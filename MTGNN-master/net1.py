from layer import *
#内层+mtgnn


class gtnet1(nn.Module):
    def __init__(self,batch_size, horizon,cycle,gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet1, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)
        self.seq_length = seq_length
        #衰减
        self.decay_rate1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.horizon = horizon
        self.cycle = cycle
        self.window = seq_length
        self.a = nn.Parameter(torch.tensor(0.7), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.inner_alpha = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        self.device='cuda:0'
        self.w = nn.Parameter(torch.rand(num_nodes, num_nodes), requires_grad=True)

        kernel_size = 7
        if dilation_exponential>1:#单步
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))#感受野 187
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to(device)


    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        # 构建图的距离矩阵
        # print('input',input.shape)# batchsize * 1 * node * 168   序列长度168
        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx,input)
                else:
                    adp = self.gc(idx,input)
            else:
                adp = self.predefined_A

        #增加时间衰减
        # 1 根据window horizon划分周期，保证待预测值是周期的最后一个（所以可以从后往前取矩阵,horizon那几个补零矩阵）；开头的矩阵可能是不完整的，不要？
        # 计算需要添加的零矩阵数量
        num_zeros_to_add = self.horizon % self.cycle
        # 在距离矩阵后面添加零矩阵
        adp_padded = F.pad(adp, (0, 0, 0, 0, 0, num_zeros_to_add))
        # print("adp_padded shape:",adp_padded.shape)

        # 计算需要从前面删除的矩阵数量
        num_matrices_to_remove = adp_padded.shape[1] % self.cycle
        # 从距离矩阵前面删除矩阵
        adp_final = adp_padded[:, num_matrices_to_remove:, :, :]

        # 2 ，在每个周期内衰减（周期内前面的权重小，后面的权重大，公式类似decay_factor = torch.exp(-self.decay_rate * (周期长度 - t))）特别注意最后一个周期可能是不完整的
        # 创建一个与距离矩阵形状相同的张量来表示时间戳
        timestamps = torch.arange(1, adp_final.shape[1] + 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(
            self.device)  # 1-35
        # 计算每个时间戳在其所在周期内的位置
        timestamps_in_cycle = (timestamps - 1) % self.cycle + 1  # 1-7 *5
        # 计算衰减因子矩阵
        decay_factor0 = F.relu(
            torch.tanh(torch.exp(-self.decay_rate1 * (self.cycle - timestamps_in_cycle))))  # 每个周期最后一个的权重是1
        decay_factor = decay_factor0.clone()
        # 将后 horizon 个衰减因子矩阵变成零矩阵
        # decay_factor[:, -(self.horizon% self.cycle):, :, :] = 0#正常衰减
        # 对于后 self.cycle 个数据，左移 self.horizon % self.cycle 个，增加后几个的权重
        decay_factor[:, -self.cycle:, :, :] = torch.roll(decay_factor[:, -self.cycle:, :, :],shifts=-(self.horizon % self.cycle), dims=1)
        # 清空horizon个
        decay_factor[:, -(self.horizon % self.cycle):, :, :] = 0
        # 将衰减因子与距离矩阵进行元素级别的乘法,得到衰减后的距离矩阵
        adp_decay = adp_final * decay_factor

        # 3 每个周期融合为一个距离矩阵
        # 假设 selected_timestamps 是你想要选择的时间戳的列表
        # 选择前面的元素，每隔 x 选一个
        front_elements = [i for i in range(1, self.cycle - ((self.cycle // 10) + 2), (self.cycle // 10) + 2)]
        # 选择周期的最后n个元素
        last_elements = [i for i in range(self.cycle - ((self.cycle // 10) + 2), self.cycle + 1)]
        # 合并前面选择的元素和周期的最后四个元素
        selected_elements = front_elements + last_elements
        # 将选定的元素转换为 PyTorch 张量
        selected_timestamps = torch.tensor(selected_elements).to(self.device)
        # 计算周期数
        num_periods = adp_decay.shape[1] // self.cycle
        # 创建一个布尔张量，表示每个时间戳是否在 selected_timestamps 中
        mask = torch.isin(timestamps_in_cycle, selected_timestamps)

        # 使用布尔索引来选择特定的时间戳的衰减因子
        selected_decay_factor = decay_factor[mask]
        # print("selected_decay_factor shape:",selected_decay_factor.shape)  #25
        # 重新调整 selected_decay_factor 的形状
        selected_decay_factor = selected_decay_factor.view(decay_factor.shape[0], num_periods,len(selected_timestamps),decay_factor.shape[2],decay_factor.shape[3])  # 1*5*4*1*1
        # print("selected_decay_factor change shape:", selected_decay_factor.shape)#1*5*5*1*1
        # 计算每个周期被选择的衰减因子的和
        sum_selected_decay_factor = torch.sum(selected_decay_factor, dim=2, keepdim=True)  # 1*5*1*1*1
        # print("sum_selected_decay_factor shape:", sum_selected_decay_factor.shape)
        # 去掉 sum_selected_decay_factor 的第三的维度
        sum_selected_decay_factor = torch.squeeze(sum_selected_decay_factor, dim=2)

        # 扩展mask到和adp_decay相同的形状
        mask = mask.expand_as(adp_decay)
        # 使用布尔索引来选择特定的时间戳的衰减距离矩阵
        selected_adp_decay = adp_decay[mask]
        # 重新调整 selected_adp_decay 的形状
        selected_adp_decay = selected_adp_decay.view(adp_decay.shape[0], num_periods, len(selected_timestamps),adp_decay.shape[2], adp_decay.shape[3])  # 4*5*4*8*8
        sum_selected_adp_decay = torch.sum(selected_adp_decay, dim=2, keepdim=True)  # 4*5*1*8*8
        # 去掉 sum_selected_adp_decay 的第三维
        sum_selected_adp_decay = torch.squeeze(sum_selected_adp_decay, dim=2)  # 4*5*8*8

        # 获取每个周期最后一个距离矩阵,计算最后的距离矩阵
        '''final_adp = (adp_decay[:, self.cycle - 1::self.cycle, :, :] + sum_selected_adp_decay) / (1 + sum_selected_decay_factor)  # adp_decay[:, self.cycle - 1::self.cycle, :, :]是周期的最后一个'''
        final_adp = (self.inner_alpha * adp_final[:, self.cycle - 1::self.cycle, :, :] + (1 - self.inner_alpha) * torch.tanh(sum_selected_adp_decay))

        # 4 周期距离矩阵融合
        final_adp1=torch.mean(final_adp, dim=1, keepdim=True)

        final_adp1=torch.squeeze(final_adp1, dim=1)#4*8*8
        final_adp1 = torch.mean(final_adp1, dim=0)
        final_adp1 = final_adp1 * self.w
        final_adp1 = F.relu(final_adp1)
        # 5 变成邻接矩阵
        """变成邻接矩阵"""

        mask = torch.zeros(self.num_nodes, self.num_nodes ).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = (final_adp1 + torch.rand_like(final_adp1) * 0.001).topk(int(self.num_nodes*0.6), 1,largest=False)#每行最小的n个。ex4,solar之前定的20,traffic40?
        mask.scatter_(1, t1, s1.fill_(1))
        adp = final_adp1 * mask

        # 如果序列长度小于感受野大小，则进行填充
        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))


        # 开始卷积层
        x = self.start_conv(input)
        """
        print('input.shape',input.shape)# batchsize * 1 * node * 187感受野
        print('x.shape', x.shape)# batchsize * 16 * node * 187感受野
        """
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x #batchsize * 16 * node * 感受野187
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)#batchsize * 16 * node * 181
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
