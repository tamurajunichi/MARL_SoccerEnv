from torch import nn, cat
import torch.nn.functional as F

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim,1024)
		self.l2 = nn.Linear(1024,512)
		self.l3 = nn.Linear(512,256)
		self.l4 = nn.Linear(256,128)
		self.l5 = nn.Linear(128,action_dim)

		nn.init.normal_(self.l1.weight.data,std=0.01)
		nn.init.normal_(self.l2.weight.data,std=0.01)
		nn.init.normal_(self.l3.weight.data,std=0.01)
		nn.init.normal_(self.l4.weight.data,std=0.01)
		nn.init.normal_(self.l5.weight.data,std=0.01)
	
		nn.init.zeros_(self.l1.bias.data)
		nn.init.zeros_(self.l2.bias.data)
		nn.init.zeros_(self.l3.bias.data)
		nn.init.zeros_(self.l4.bias.data)
		nn.init.zeros_(self.l5.bias.data)

	def forward(self, state):
		out = F.leaky_relu(self.l1(state))
		out = F.leaky_relu(self.l2(out))
		out = F.leaky_relu(self.l3(out))
		out = F.leaky_relu(self.l4(out))
		return self.l5(out)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, td3):
		super(Critic, self).__init__()
		self.td3 = td3

		#Q1
		self.l1 = nn.Linear(state_dim+action_dim,1024)
		self.l2 = nn.Linear(1024,512)
		self.l3 = nn.Linear(512,256)
		self.l4 = nn.Linear(256,128)
		self.l5 = nn.Linear(128,1)

		nn.init.normal_(self.l1.weight.data,std=0.01)
		nn.init.normal_(self.l2.weight.data,std=0.01)
		nn.init.normal_(self.l3.weight.data,std=0.01)
		nn.init.normal_(self.l4.weight.data,std=0.01)
		nn.init.normal_(self.l5.weight.data,std=0.01)

		nn.init.zeros_(self.l1.bias.data)
		nn.init.zeros_(self.l2.bias.data)
		nn.init.zeros_(self.l3.bias.data)
		nn.init.zeros_(self.l4.bias.data)
		nn.init.zeros_(self.l5.bias.data)

		#Q2
		if td3 == True:
			self.l6 = nn.Linear(state_dim+action_dim,1024)
			self.l7 = nn.Linear(1024,512)
			self.l8 = nn.Linear(512,256)
			self.l9 = nn.Linear(256,128)
			self.l10 = nn.Linear(128,1)

			nn.init.normal_(self.l6.weight.data,std=0.01)
			nn.init.normal_(self.l7.weight.data,std=0.01)
			nn.init.normal_(self.l8.weight.data,std=0.01)
			nn.init.normal_(self.l9.weight.data,std=0.01)
			nn.init.normal_(self.l10.weight.data,std=0.01)

			nn.init.zeros_(self.l6.bias.data)
			nn.init.zeros_(self.l7.bias.data)
			nn.init.zeros_(self.l8.bias.data)
			nn.init.zeros_(self.l9.bias.data)
			nn.init.zeros_(self.l10.bias.data)

	def forward(self, state, action):
		inp = cat((state,action),1)

		q1 = F.leaky_relu(self.l1(inp))
		q1 = F.leaky_relu(self.l2(q1))
		q1 = F.leaky_relu(self.l3(q1))
		q1 = F.leaky_relu(self.l4(q1))
		q1 = self.l5(q1)

		q2 = F.leaky_relu(self.l6(inp))
		q2 = F.leaky_relu(self.l7(q2))
		q2 = F.leaky_relu(self.l8(q2))
		q2 = F.leaky_relu(self.l9(q2))
		q2 = self.l10(q2)

		return q1,q2

	def Q1(self, state, action):
		inp = cat((state, action), 1)

		q1 = F.leaky_relu(self.l1(inp))
		q1 = F.leaky_relu(self.l2(q1))
		q1 = F.leaky_relu(self.l3(q1))
		q1 = F.leaky_relu(self.l4(q1))
		q1 = self.l5(q1)
		return q1
