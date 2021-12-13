from helper import *
from data_loader import *

# sys.path.append('./')
from model.models import JointModel
from model.kg_node_embedding import KGNodeEmbedding
import itertools

class Runner(object):
    def load_data(self):
        """
        Reading in raw triples and converts it into a standard format.

        Parameters
        ----------
        self.p.dataset:         Takes in the name of the dataset (FB15k-237)

        Returns
        -------
        self.ent2id:            Entity to unique identifier mapping
        self.id2rel:            Inverse mapping of self.ent2id
        self.rel2id:            Relation to unique identifier mapping
        self.num_ent:           Number of entities in the Knowledge graph
        self.num_rel:           Number of relations in the Knowledge graph
        self.embed_dim:         Embedding dimension used
        self.data['train']:     Stores the triples corresponding to training dataset
        self.data['valid']:     Stores the triples corresponding to validation dataset
        self.data['test']:      Stores the triples corresponding to test dataset
        self.data_iter:		The dataloader for different data splits
        self.target_index:		Stores the target nodes for predicting
        self.source_index:		Stores the source nodes for predicting

        """

        org = OrderedSet()
        author = OrderedSet()
        keyword = OrderedSet()
        venue = OrderedSet()
        paper = OrderedSet()
        ent_set = OrderedSet()
        rel_set = OrderedSet()
        city = OrderedSet()
        country = OrderedSet()
        kbs_tripple = []
        examples ={}

        #知识图谱中，先按set单独编码，然后每种得到整个实体编码
        #将实体按种类次序拼接得到其在知识图谱实体中的index,将上述编码作为实体编码矩阵


        if self.p.dataset == 'OrgPaper':
            pass #之后加入多个数据集的时候再晚上


        #1. 读入层间的链接信息
        PAO = json.load(open('./data/{}/{}'.format(self.p.dataset, 'coauthor.json'), 'r', encoding='utf-8'))
        org_co = {}#其中的每个元素为({'(org1,org2)':[paper1,paper2,paper3]})
        for key,item in PAO.items():
            if len(item['org_cooperate'])>1:
                for i in itertools.combinations(item['org_cooperate'], 2):
                    if i not in org_co:
                        if ((i[-1],i[0]) not in org_co):
                            org_co[i] = [key]
                        else:
                            org_co[(i[-1],i[0])].append(key)
                    else:
                        org_co[i].append(key)

        #按组织关系划分初始数据(50%)、训练集(30%)、验证集(10%)、测试集(10%)
        #这部分数据保留层间链接，加入kbs中，记录AP信息
        hpaper_count = int(len(org_co)*self.p.hypergraph_split)
        org_cooperate_known = []#里面的元素为(org1,org2)
        org_cooperate_unknown = []
        count = 0
        for key,item in org_co.items():
            count += 1
            if count > hpaper_count:
                org_cooperate_known.append(key)
            else:
                org_cooperate_unknown.append(key)
        # 进一步划分org_cooperate_unknown为训练集(30%)、验证集(10%)、测试集(10%)
        train_num = int(0.6*len(org_cooperate_unknown))
        random.shuffle(org_cooperate_unknown)
        examples['train'] = org_cooperate_unknown[:train_num]
        examples['valid'] = org_cooperate_unknown[train_num:train_num+ int(0.2*len(org_cooperate_unknown))]
        examples['test'] = org_cooperate_unknown[train_num+ int(0.2*len(org_cooperate_unknown)):]


        #将已知org_cooperate_known中的AP加入KBS
        def set_include_item(org_cooperate_set,org_cooperate_unknown):
            for (org1,org2) in org_cooperate_unknown:
                if org1 in org_cooperate_set and org2 in org_cooperate_set:
                    return True
            return False
        for key,item in PAO.items():
            if set_include_item(item["org_cooperate"], org_cooperate_unknown):#检查item["org_cooperate"] 是否包含了org_cooperate_unknown中的任一元素
                pass
            else:
                for each_author in item["author_cooperate"]:
                    kbs_tripple.append((key, 'paper/author/write', each_author))
                    author.add(each_author)
                    paper.add(key)
        rel_set.add('paper/author/write')

        #2. 读入kg实体信息
        for line in open('./data/{}/{}'.format(self.p.dataset, 'kb.txt'), 'r', encoding='utf-8'):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                if 'keywords' in sub:
                    keyword.add(sub)
                if 'keywords' in obj:
                    keyword.add(obj)
                if 'venue' in sub:
                    venue.add(sub)
                if 'venue' in obj:
                    venue.add(obj)
                if 'paper' in sub:
                    paper.add(sub)
                if 'paper' in obj:
                    paper.add(obj)
                if 'org' in sub:
                    org.add(sub)
                if 'org' in obj:
                    org.add(obj)
                if 'country' in sub:
                    country.add(sub)
                if 'country' in obj:
                    country.add(obj)
                if 'city' in sub:
                    city.add(sub)
                if 'city' in obj:
                    city.add(obj)
                rel_set.add(rel)
                kbs_tripple.append((sub, rel, obj))



        self.num_rel = len(rel_set)
        print('kg中关系数',self.num_rel)
        print('kg中关键词数', len(keyword))
        print('kg中期刊', len(venue))
        print('kg中组织数', len(org))
        print('kg中作者数', len(author))
        self.org_kg_index = torch.LongTensor([idx for idx in range(len(org))]).to(self.device)
        self.author_kg_index = torch.LongTensor([idx+len(org) for idx in range(len(author))]).to(self.device)




        #4. 读入组织超图
        #分为两部分[已知的人，未知的人],已知部分id与知识图谱部分一致
        #分为两部分[已知的组织，未知的组织]
        #对于知识图谱中已有的组织及人物，取其id(从1开始到指定数量)获取表征
        #对知识图谱中没有的组织及人物，初始化为全0
        # hyper_index: 按人编一个，按组织编一个。把两部分按顺序合起来（第二种类型序号接着第一种的最后一个）形成超图的所有节点
            # 预测时取表征时，取组织表征相乘(日后可考虑取超边表征)
        OA = json.load(open('./data/{}/{}'.format(self.p.dataset, 'org_author.json'), 'r', encoding='utf-8'))
        hyper_edge = []
        for (org_name,author_group) in OA.items():
            org.add(org_name)
            for each_author in author_group:
                author.add(each_author)
            author_group.append(org_name)
            hyper_edge.append(author_group)
        self.extra_org_num = len(org) - self.org_kg_index.size(0)
        self.extra_author_num = len(author) - self.author_kg_index.size(0)

        #5. 编号
        #按次序合并entity
        ent_set.update(org)
        ent_set.update(author)
        ent_set.update(keyword)
        ent_set.update(venue)
        ent_set.update(paper)
        ent_set.update(city)
        ent_set.update(country)
        self.node_set_length = torch.LongTensor([len(org),len(author),len(keyword),len(venue),len(paper),len(city),len(country)]).to(self.device)
        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})
        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}
        self.p.num_ent = len(self.ent2id)


        #6.1 对超边进行编码，处理为能输入图形的形式
        hyper_edge_id = []
        hyper_node_id = []
        for index,edge in enumerate(hyper_edge):
            for node in edge:
                hyper_edge_id.append(index)
                hyper_node_id.append(self.ent2id[node])
        self.hyper_edge = torch.LongTensor([hyper_node_id,hyper_edge_id]).to(self.device)

        #6.2 对kb进行编码，处理为能输入图形的形式
        for index,tripple in enumerate(kbs_tripple):
            kbs_tripple[index] = (self.ent2id[tripple[0]],self.rel2id[tripple[1]],self.ent2id[tripple[2]])
        #Constructs the adjacency matrix for GCN
        self.kb_edge_index, self.kb_edge_type = [], []
        # 使用pyG时，edge_index要指明入边和出边
        for sub, rel, obj in kbs_tripple:
            self.kb_edge_index.append((sub, obj))
            self.kb_edge_type.append(rel)
        # Adding inverse edges
        for sub, rel, obj in kbs_tripple:
            self.kb_edge_index.append((obj, sub))
            self.kb_edge_type.append(rel + self.num_rel)
        self.kb_edge_index = torch.LongTensor(self.kb_edge_index).to(self.device).t()
        self.kb_edge_type = torch.LongTensor(self.kb_edge_type).to(self.device)


        #把训练集、验证集、测试集都处理为{'source':['target1','target2',...]}的格式
        s_to_o = {}
        s_to_o['train'] = {}
        s_to_o['valid'] = {}
        s_to_o['test'] = {}
        for split_set in ['train','valid','test']:
            for (s,o) in examples[split_set]:
                if self.ent2id[s] in s_to_o[split_set]:
                    s_to_o[split_set][self.ent2id[s]].add(self.ent2id[o])
                else:
                    s_to_o[split_set][self.ent2id[s]] = set()
                    s_to_o[split_set][self.ent2id[s]].add(self.ent2id[o])
                if self.ent2id[o] in s_to_o[split_set]:
                    s_to_o[split_set][self.ent2id[o]].add(self.ent2id[s])
                else:
                    s_to_o[split_set][self.ent2id[o]] = set()
                    s_to_o[split_set][self.ent2id[o]].add(self.ent2id[s])
        #以前的格式是s\s的all_label
        #转化成单个样本,s\o\s的all_label   若只要正负样本则为s\o
        s_o_all = {}
        s_o_all['train'] = {} #{(s,o):[o1,o2,o3,....]}
        s_o_all['valid'] = {}
        s_o_all['test'] = {}
        for split_set in ['train','valid','test']:
            for s,o_list in s_to_o[split_set].items():
                for o in o_list:
                    s_o_all[split_set][(s,o)] = o_list
        print(s_o_all['test'])



        # 7. 获取训练数据
        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
                return DataLoader(
                    dataset_class(s_o_all[split], self.p, [i for i in range(len(org))]),
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=max(0, self.p.num_workers),
                    collate_fn=dataset_class.collate_fn
                )

        self.data_iter = {
                    'train': get_data_loader(TrainDataset, 'train', self.p.batch_size),
                    'valid': get_data_loader(TestDataset, 'valid', self.p.batch_size),
                    'test': get_data_loader(TestDataset, 'test', self.p.batch_size)
                }

    def __init__(self, params):
        """
        Constructor of the runner class

        Parameters
        ----------
        params:         List of hyper-parameters of the model

        Returns
        -------
        Creates computational graph and optimizer

        """
        self.p = params  # 输入的参数，parser.parse_args()生成
        self.logger = get_logger(self.p.store_name, self.p.log_dir, self.p.config_dir)

        self.logger.info(vars(self.p))  # vars(self.p) 把args转换成字典

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()
        #这一部分还得改
        self.model = JointModel(self.node_set_length, self.kb_edge_index, self.kb_edge_type, self.num_rel, self.org_kg_index, self.author_kg_index, self.extra_org_num, self.extra_author_num, self.hyper_edge, self.p)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)

    def fit(self):
        """
        Function to run training and evaluation of model

        Parameters
        ----------

        Returns
        -------
        """
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        save_path = os.path.join('./checkpoints', self.p.store_name.replace(':', ''))

        if self.p.restore:  # 是否载入训练好的模型
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')
        else:
            kill_cnt = 0
            for epoch in range(self.p.max_epochs):
                train_loss = self.run_epoch(epoch, val_mrr)
                val_results = self.evaluate('valid', epoch)  # 每一个epoch后，在valid上验证结果

                if val_results['mrr'] > self.best_val_mrr:
                    self.best_val = val_results
                    self.best_val_mrr = val_results['mrr']
                    self.best_epoch = epoch
                    self.save_model(save_path)
                    kill_cnt = 0
                else:
                    kill_cnt += 1
                    if kill_cnt % 10 == 0 and self.p.gamma > 5:
                        self.p.gamma -= 5
                        self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                    if kill_cnt > 25:
                        self.logger.info("Early Stopping!!")
                        break

                self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(epoch, train_loss,
                                                                                                 self.best_val_mrr))
        self.logger.info('Loading best model, Evaluating on Test data')
        self.load_model(save_path)
        test_results = self.evaluate('test', 0)

    def run_epoch(self, epoch, val_mrr=0):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count

        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
       #self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):
            source, label = batch
            source = source.to(self.device)
            label = label.to(self.device)
            source = source.squeeze(-1)#[batch,1]变为[batch]
            pred = self.model.forward(source)
            loss = self.model.loss(pred, label)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if step % 100 == 0:
                self.logger.info(
                    '[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses),
                                                                              self.best_val_mrr,
                                                                              self.p.store_name))  # 这里一直显示的是到目前epoch为止最好的Val MRR

        loss = np.mean(losses)
        self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
        return loss

    def evaluate(self, split, epoch):
        """
        Function to evaluate the model on validation or test set

        Parameters
        ----------
        split: (string) If split == 'valid' then evaluate on the validation set, else the test set
        epoch: (int) Current epoch count

        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """

        results  = self.predict(split=split)
        self.logger.info \
            ('[Epoch {} {}]: MRR: {:.5}'.format(epoch, split, results['mrr']))
        return results

    def predict(self,split='valid'):
        """
        Function to run model evaluation for a given mode

        Parameters
        ----------
        split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
        mode: (string):		Can be 'head_batch' or 'tail_batch'

        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """

        self.model.eval()
        results = {}
        with torch.no_grad():
            train_iter = iter(self.data_iter[split])
            for step, batch in enumerate(train_iter):
                source, label, obj = batch
                source = source.to(self.device)
                label = label.to(self.device)
                obj = obj.to(self.device)
                source = source.squeeze(-1)  # [batch,1]变为[batch]
                obj = obj.squeeze(-1)
                pred = self.model.forward(source)

                b_range = torch.arange(pred.size()[0], device=self.device)

                target_pred = pred[b_range, obj]  # 取batch 中每一个 obj对应的pred的值
                pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000,
                                   pred)  # (B,target_nodes),其余全为-1000，只有label地方的为pred的值
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]  # 将pred从小到大排列，取label对应的index
                print('ranks',ranks)

                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.mean(ranks).item() + results.get('mr',    0.0)
                results['mrr'] = torch.mean(1.0 / ranks).item() + results.get('mrr',   0.0)
                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)])

            if step % 100 == 0:
                self.logger.info('[{} Step {}]\t{}'.format(split.title(), step, self.p.store_name))

        results['count'] = results['count']/(step+1)
        results['mr'] = results['mr']/(step+1)
        results['mrr'] = results['mrr']/(step+1)
        for k in range(10):
            results['hits@{}'.format(k + 1)] = results['hits@{}'.format(k + 1)]/(step+1)

        return results

    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved

        Returns
        -------
        """
        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args'	: vars(self.p)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model

        Returns
        -------
        """
        state			= torch.load(load_path)
        state_dict		= state['state_dict']
        self.best_val		= state['best_val']
        self.best_val_mrr	= self.best_val['mrr']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-store_name', default='testrun', help='Set run name for saving/restoring models')
    parser.add_argument('-dataset', default='OrgPaper', help='Dataset to use, default: OrgPaper')
    parser.add_argument('-restore', default=False, help='Restore from the previously saved model')  # 是否载入原模型，默认为False
    parser.add_argument('-logdir', dest='log_dir', default='./log/', help='Log directory')
    parser.add_argument('-config', dest='config_dir', default='./config/', help='Config directory')

    #已知超图和训练+测试比例划分设置
    parser.add_argument('-hypergraph_split', default=0.5, type=float, help='percentage of fixed hypergraph nodes')

    #模型基础设置
    parser.add_argument('-batch', dest='batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('-gamma', type=float, default=40.0, help='Margin')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch', dest='max_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('-l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('-lr', type=float, default=0.0001, help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('-opn', default='mult', help='Composition Operation to be used in CompGCN')
    parser.add_argument('-bias', action='store_true', help='Whether to use bias in the model')
    parser.add_argument('-num_workers', type=int, default=0, help='Number of processes to construct batches')
    parser.add_argument('-seed', dest='seed', default=41504, type=int, help='Seed for randomization')


    #kg编码部分
    parser.add_argument('-kg_model', default = 'compGCN', help='Graph models for encoding kbs')
    parser.add_argument('-add_inverse', default = True, help='add inverse edges for kbs')
    parser.add_argument('-num_bases', default=-1, type=int, help='Number of basis relation vectors to use')
    parser.add_argument('-init_dim', default=100, type=int, help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim', default=100, type=int, help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim', default=100, type=int, help='Embedding dimension to give as input to score function')#output
    parser.add_argument('-gcn_layer', default=1, type=int, help='Number of GCN Layers to use')
    parser.add_argument('-dropout', default=0.1, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop', default=0.2, type=float, help='Dropout after GCN')

    #超图编码部分
    parser.add_argument('-hyper_model', default = 'HCHA', help='Graph models for encoding hypergraph')
    parser.add_argument('-hyper_layers', default=1, help='Number of hyper layers to use')
    parser.add_argument('-HyperAtt', default=False, help='Use attention in hyper layers or not')
    parser.add_argument('-hyper_init_dim', default=100, type=int, help='Initial dimension size for hypergraph')
    parser.add_argument('-hyper_hidden', default=100, help='Hidden dimension size for hypergraph')

    args = parser.parse_args()



    if not args.restore:
        args.store_name = args.store_name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')  # log文件的命名，在这里进行了更改
    else:
        args.store_name = 'testrun_06_09_2021_191455'  # 自定义的文件名

    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = Runner(args)
    model.fit()
    print('miaomiaowangdeceshiaa')