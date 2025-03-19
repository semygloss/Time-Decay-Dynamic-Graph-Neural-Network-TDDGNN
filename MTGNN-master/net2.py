from layer import *
#外层+mtgnn


class gtnet2(nn.Module):
    def __init__(self,batch_size, horizon,cycle,gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet2, self).__init__()
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
        self.decay_rate= nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.horizon = horizon
        self.cycle = cycle
        self.window = seq_length
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

        # 创建一个与距离矩阵形状相同的张量来表示时间戳
        timestamps = torch.arange(1, adp.shape[1] + 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(self.device)
        # 计算衰减因子矩阵
        decay_factor = torch.exp(-self.decay_rate * (adp.shape[1] - timestamps))
        # print(decay_factor.shape)
        # 将衰减因子与距离矩阵进行元素级别的乘法,得到衰减后的距离矩阵
        adp_decay = adp * decay_factor
        # print('adp_decay:',adp_decay)
        # 随机生成m的值，假设m的最大值为max_m
        # max_m = 10
        # m = torch.randint(1, max_m + 1, (1,)).item()
        # print('m:',m)
        # 随机生成一个0到32-2的排列，然后取前m个元素作为时间戳索引
        # selected_timestamps = torch.randperm(adp.shape[1] - 1)[:m]
        # print('selectm:',selected_timestamps)

        # 假设 selected_timestamps 是你想要选择的时间戳的列表
        # 选择前面的元素，每隔 x 选一个
        front_elements = [i for i in range(1, self.window - 10, self.cycle)]
        # 选择周期的最后n个元素
        last_elements = [i for i in range(self.window - 9, self.window)]
        # 合并前面选择的元素和周期的最后四个元素
        selected_elements = front_elements + last_elements
        # 将选定的元素转换为 PyTorch 张量
        selected_timestamps = torch.tensor(selected_elements).to(self.device)

        # 使用 torch.index_select 来选择特定的时间戳
        selected_adp_decay = torch.index_select(adp_decay, 1, selected_timestamps.to(self.device))
        selected_decay_factor = torch.index_select(decay_factor, 1, selected_timestamps.to(self.device))
        # 计算选择的衰减距离矩阵之和，衰减因子之和
        sum_selected_adp_decay = torch.sum(selected_adp_decay, dim=1)
        sum_selected_decay_factor = torch.sum(selected_decay_factor, dim=1)
        # 计算最后的距离矩阵
        final_adp1 = (adp_decay[:, -1, :, :] + sum_selected_adp_decay) / (1 + sum_selected_decay_factor)  # adp_decay[:, -1, :, :]是样本的最后一个

        final_adp1 = torch.mean(final_adp1, dim=0)
        final_adp1 = final_adp1 * self.w
        final_adp1 = F.relu(final_adp1)
        # 5 变成邻接矩阵
        """变成邻接矩阵"""
        #print(final_adp1)
        """用mask"""
        # final_adp1 = torch.where(final_adp1 < 0.53, 1.0, 0.0)#这里可以改好点
        # final_adp1=F.normalize(final_adp1, p=1, dim=1)  # 4*8*8
        # g_decay = final_adp1 * self.w
        # adp = F.relu(g_decay)

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
