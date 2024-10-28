
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()

        self.BlockLayer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=(3,3), padding=(1,1)),
                    
            nn.BatchNorm2d(num_features=out_channels),
        
            nn.ReLU(),
        
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), padding=(1,1)),

            nn.BatchNorm2d(num_features=out_channels)
                
        )

        self.batch_Norm = nn.BatchNorm2d(num_features=out_channels )

        self.relu = nn.ReLU()

        self.connReq = True

        self.skipConn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=(1,1))

        if in_channels == out_channels and stride == (1,1):
            self.connReq = False

        else:
            self.connReq = True


    def forward(self, input_tensor):

        output = self.BlockLayer(input_tensor)
        skipcon = input_tensor
        
        if self.connReq:
            skipcon = self.skipConn(input_tensor)
        
        skipcon = self.batch_Norm(skipcon)    

        output = output + skipcon
        output = self.relu(output)

        return output



class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.ResLayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, stride=(2,2), kernel_size=(7,7)),

            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

            ResBlock(in_channels=64, out_channels=64, stride=(1,1)),
            ResBlock(in_channels=64, out_channels=128, stride=(2,2)),
            ResBlock(in_channels=128, out_channels=256, stride=(2,2)),
            ResBlock(in_channels=256, out_channels=512, stride=(2,2)),

            nn.AvgPool2d(kernel_size=(10,10)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=2),
            nn.Sigmoid()

        )

        

    def forward(self, input_tensor):
        output = self.ResLayer(input_tensor)


        return output