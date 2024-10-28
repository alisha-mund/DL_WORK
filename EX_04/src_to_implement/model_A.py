from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.convLr1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=3, padding=1)
        self.convLr2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=1, kernel_size=3, padding=1)
        self.batNor = nn.BatchNorm2d(num_features=out_channels)
        self.rel = nn.ReLU()

        # if stride != 1 or in_channels != out_channels:
            # self.skipConn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=1)
        self.skipConn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=1),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, input_tensor):
        output = self.convLr1(input_tensor)
        output = self.batNor(output)
        output = self.rel(output)
        output = self.convLr2(output)
        output = self.batNor(output)

        skipcon = self.skipConn(input_tensor)
        # skipcon = self.batNor(skipcon)

        output = skipcon + output
        output = self.rel(output)

        return output



class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # self.convLr1 = nn.Conv2d(in_channels=3, out_channels=64, stride=(2,2), kernel_size=(7,7), padding=(3,3))
        # self.batNor = nn.BatchNorm2d(num_features=64)
        # self.rel = nn.ReLU()
        # self.MPo = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        # self.ResB1 = ResBlock(in_channels=64, out_channels=64, stride=(1,1))
        # self.ResB2 = ResBlock(in_channels=64, out_channels=128, stride=(2,2))
        # self.ResB3 = ResBlock(in_channels=128, out_channels=256, stride=(2,2))
        # self.ResB4 = ResBlock(in_channels=256, out_channels=512, stride=(2,2))
        # self.APo = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # self.flat = nn.Flatten()
        # self.fullConn = nn.Linear(in_features=512, out_features=2)
        # self.sig = nn.Sigmoid()

        self.convLr1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.batNor = nn.BatchNorm2d(num_features=64)
        self.rel = nn.ReLU()
        self.MPo = nn.MaxPool2d(kernel_size=3, stride=2)
        self.ResB1 = ResBlock(in_channels=64, out_channels=64, stride=1)
        self.ResB2 = ResBlock(in_channels=64, out_channels=128, stride=2)
        self.ResB3 = ResBlock(in_channels=128, out_channels=256, stride=2)
        self.ResB4 = ResBlock(in_channels=256, out_channels=512, stride=2)
        self.APo = nn.AdaptiveAvgPool2d(output_size=1)
        self.flat = nn.Flatten()
        self.fullConn = nn.Linear(in_features=512, out_features=2)
        self.sig = nn.Sigmoid()

    def forward(self, input_tensor):
        output = self.convLr1(input_tensor)
        output = self.batNor(output)
        output = self.rel(output)
        output = self.MPo(output)

        output = self.ResB1(output)
        output = self.ResB2(output)
        output = self.ResB3(output)
        output = self.ResB4(output)

        output = self.APo(output)
        output = self.flat(output)
        output = self.fullConn(output)
        output = self.sig(output)

        return output