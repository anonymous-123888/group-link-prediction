from helper import *
from model.hypergraph_layer import *
from model.kg_layer import *
from model.kg_node_embedding import *


class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p		= params
		self.act	= torch.tanh
		self.bceloss	= torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)

class JointModel(BaseModel):
	#读入数据时，对超图中的实体按类型重新编码
	#hyper_index: 按人编一个，按组织编一个。把两部分按顺序合起来（第二种类型序号接着第一种的最后一个）形成超图的所有节点
	#预测时取表征时，取组织表征相乘(日后可考虑取超边表征)
	def __init__(self, node_set_length, edge_index, edge_type, num_rel, org_kg_index, author_kg_index, extra_org_num, extra_author_num, hyper_edge, params=None):#初始化时需要传入构造知识图及超图所需的全部数据
		super(JointModel, self).__init__(params)
		self.node_set_length = node_set_length
		self.kg_edge_type = edge_type
		self.kg_edge_index = edge_index
		self.num_rel = num_rel
		self.org_kg_index = org_kg_index
		self.author_kg_index = author_kg_index
		self.hyper_edge = hyper_edge
		self.extra_org_num = extra_org_num
		self.extra_author_num = extra_author_num
		self.drop = torch.nn.Dropout(self.p.hid_drop)
		self.kg_embedding_layer = KGNodeEmbedding(self.node_set_length, self.p)

		if self.p.kg_model == 'compGCN':
			self.KGBase = CompGCN(self.kg_edge_index, self.kg_edge_type, num_rel, params=self.p)

		if self.p.hyper_model == 'HCHA':
			self.HyperBase = HCHA(params=self.p)

	def forward(self,  source):
		kg_node_emb = self.kg_embedding_layer()
		org_emb_kg, author_emb_kg	= self.KGBase.forward_base(kg_node_emb, self.org_kg_index, self.author_kg_index, self.drop, self.drop)#使用训练好的图谱，得到组织及作者表征
		#print('org_out',org_emb_kg.size())
		#print('author_out', author_emb_kg.size())
		extra_org_emb = torch.zeros(self.extra_org_num,self.p.hyper_init_dim).float().cuda()
		#print('extra_org_emb', extra_org_emb.size())
		extra_author_emb = torch.zeros(self.extra_author_num,self.p.hyper_init_dim).float().cuda()
		hyper_org = torch.cat([org_emb_kg,extra_org_emb],dim=0)
		#print('hyper_org', hyper_org.size(), hyper_org)
		hyper_author = torch.cat([author_emb_kg,extra_author_emb],dim=0)
		hyper_node = torch.cat([hyper_org, hyper_author],dim=0)

		x = self.HyperBase.forward_base(hyper_node, self.hyper_edge)

		#print('生成的节点表征',x.size(),x)

		# 损失函数之后再编写，先看看能不能正确生成x
		test_source_emb = torch.index_select(x, 0, source) #(n_source,emb)
		#print('源节点形状', test_source_emb.size(), test_source_emb)
		#target_index 为所有组织节点在超图中的序号,即[-org_num,:]
		all_target_emb = torch.index_select(x, 0, torch.LongTensor([idx for idx in range(hyper_org.size(0))]).cuda()) #(n_target,emb)

		score = torch.mm(test_source_emb,all_target_emb.t())# (n_source,n_target)
		score = torch.sigmoid(score)
		#print('score',score.size(),score)

		return score

class OnlyKGModel(BaseModel):
	#读入数据时，对超图中的实体按类型重新编码
	#hyper_index: 按人编一个，按组织编一个。把两部分按顺序合起来（第二种类型序号接着第一种的最后一个）形成超图的所有节点
	#预测时取表征时，取组织表征相乘(日后可考虑取超边表征)
	def __init__(self, node_set_length, edge_index, edge_type, num_rel, org_kg_index, params=None):#初始化时需要传入构造知识图及超图所需的全部数据
		super(OnlyKGModel, self).__init__(params)
		self.node_set_length = node_set_length
		self.kg_edge_type = edge_type
		self.kg_edge_index = edge_index
		self.num_rel = num_rel
		self.org_kg_index = org_kg_index
		self.drop = torch.nn.Dropout(self.p.hid_drop)
		self.kg_embedding_layer = KGNodeEmbedding(self.node_set_length, self.p)

		if self.p.kg_model == 'compGCN':
			self.KGBase = CompGCN(self.kg_edge_index, self.kg_edge_type, num_rel, params=self.p)

	def forward(self,  source):
		kg_node_emb = self.kg_embedding_layer()
		org_emb_kg, all_nodes_emb	= self.KGBase.forward_base(kg_node_emb, self.org_kg_index, self.drop, self.drop)#使用训练好的图谱，得到组织及作者表征

		test_source_emb = torch.index_select(org_emb_kg, 0, source) #(n_source,emb)
		all_target_emb = org_emb_kg

		score = torch.mm(test_source_emb,all_target_emb.t())# (n_source,n_target)
		score = torch.sigmoid(score)
		#print('score',score.size(),score)

		return score