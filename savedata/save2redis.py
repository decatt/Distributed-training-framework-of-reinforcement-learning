import datetime
import redis,os
import numpy as np
import logging


class Save2redis(object):
    def __init__(self,
            version_id:int,
            capacity:int,
            model_type='onnx',
            check_input=True,
            check_model_pth=True,
            check_tensorboard_pth=True,
            host='127.0.0.1', # 'localhost'
            port=6397,# multi port for hight communication
            db=0,
            password=None) -> None:
        super(self,Save2redis).__init__()

        """
            {
                STATE : [S1,S2,S3...],
                ACTION:[A1,A2,A3...],
                LOG_PROBS:[P1,P2,P3...],
                MASKS:[M1,M2,M3...],
                VALUES:[VAL1,VAL2,VAL3...],
                REWARDS:[R1,R2,R3...]
                DS:[D1,D2,D3...]
                VERSION : [V1, V2, V3...],
            }

        """
        self.capacity=capacity
        self.model_type=model_type
        self.check_input=check_input
        self.check_model_pth=check_model_pth
        self.check_tensorboard_pth=check_tensorboard_pth
        self.version_name=self._get_version_format(version_id)
        self.host=host
        self.port=port
        pool = redis.ConnectionPool(host=host, port=port,db=db,password=password)
        self.conn = redis.Redis(connection_pool=pool)
        
    def __len__(self,version_id=None):
        return self.capacity



    def save2json(self):
        pass
    def loadjson(self,path):
        pass


    def sample(self,mode='random',names=['state','actions','log_probs','masks','values','rewards','ds','version']):
        self._check_length()

        length=self.conn.llen('state')
        if mode=='random':
            index=np.random.randint(length,size=1)
        else:
            index=-1
        state,actions,log_probs,masks,values,rewards,ds,version=self.conn.lindex(names[0],index),\
            self.conn.lindex(names[1],index),\
            self.conn.lindex(names[2],index),\
            self.conn.lindex(names[3],index),\
            self.conn.lindex(names[4],index),\
            self.conn.lindex(names[5],index),\
            self.conn.lindex(names[6],index),\
            self.conn.lindex(names[7],index)

        # TIME COMSUMING?
        v = self.conn.lrem(names[0], state, 1)
        v = self.conn.lrem(names[1], actions, 1)
        v = self.conn.lrem(names[2], log_probs, 1)
        v = self.conn.lrem(names[3], masks, 1)
        v = self.conn.lrem(names[4], values, 1)
        v = self.conn.lrem(names[5], rewards, 1)
        v = self.conn.lrem(names[6], ds, 1)
        v = self.conn.lrem(names[7], version, 1)

        # return state,actions,log_probs,masks,values,rewards,ds,version
        return np.fromstring(state,dtype=np.float32),\
            np.fromstring(actions,dtype=np.float32),\
                np.fromstring(log_probs,dtype=np.float32),\
                    np.fromstring(masks,dtype=np.float32),\
                        np.fromstring(values,dtype=np.float32),\
                            np.fromstring(rewards,dtype=np.float32),\
                                np.fromstring(ds,dtype=np.float32),\
                                    np.fromstring(version,dtype=np.int32)


    def push(self,inputs,names=['state','actions','log_probs','masks','values','rewards','ds','version']):
        assert len(inputs)==8
        for i,input_ in enumerate(inputs):
            self.push_single(input_,names[i])


    def push_single(self,input_,name):
        existing_length=self.conn.llen(name)
        if existing_length<self.capacity:
            v=self.conn.lpush(name,input_)
        elif existing_length==self.capacity:
            self.conn.rpop(name)
        else:
            assert False and f"Existing length={existing_length} of the list is beyond capacity={self.capacity} of redis"

            
    def _check_length(self,names=['state','actions','log_probs','masks','values','rewards','ds','version']):
        lengths=[self.conn.llen(name) for name in names]
        for i in range(len(names)-1):
            assert lengths[i]==lengths[i+1] and f"In redist the length of {names[i]} is not equal to {names[i+1]}" and f"length_{names[i]}={lengths[i]} ; length_{names[i+1]}={lengths[i+1]}"

    def update_model_path(self,model_pth):
        NOW=datetime.now().strftime("%Y_%m_%d_%H_%M")
        pass
    def update_tensorboard_path(self,tensorboard_pth):
        """
            FORMAT: ("%Y_%m_%d_%H_%M")
        """
        NOW=datetime.now().strftime("%Y_%m_%d_%H_%M")
        pass
    
    def get_replay_buffer(self,batch_size,version_id=None):
        version_name=self.version_name if version_id==None else self._get_version_format(version_id)
        pass
    
    def get_model_path(self,version_id=None):
        version_name=self.version_name if version_id==None else self._get_version_format(version_id)
        pass

    def get_tensorboar_path(self,version_id=None):
        version_name=self.version_name if version_id==None else self._get_version_format(version_id)
        pass

    def get_latest_model_pth(self,version_id=None):
        version_name=self.version_name if version_id==None else self._get_version_format(version_id)
        NOW=datetime.now().strftime("%Y_%m_%d_%H_%M")
        pass

    def get_latest_tensorboard_pth(self,version_id=None):
        version_name=self.version_name if version_id==None else self._get_version_format(version_id)
        pass
    def delete_certain_version(self,version_id=None):
        version_name=self.version_name if version_id==None else self._get_version_format(version_id)
        pass


    @staticmethod
    def _get_version_format(version_id):
        version_name=f"version_PPO_{version_id}"
        return version_name
    @staticmethod
    def _check_model_path(model_pth,default_type='onnx'):
        assert model_pth.endswith('.'+default_type)
        assert os.path.isfile(model_pth) and "FileError: model path must be a file"
        assert isinstance(model_pth,str) and 'InputError: model path should be str type'
    @staticmethod
    def _check_tensorboard_path(tensorboard_dir):
        assert os.path.isfile(tensorboard_dir) or os.path.isdir(tensorboard_dir)
        assert isinstance(tensorboard_dir,str) and 'InputError: tensorboard_dir should be str type'
    @staticmethod
    def _check_input_data(input_one:list):
        """
            every input in inputs contains:
                observation: np.ndarray
                action: np.ndarray or int or float
                reward: float
                next_observation: np.ndarray
                info: dict
                network_onnx_path: str
                tensorboard_dir: str

        """
        if isinstance(input_one,(list,tuple)):
            assert len(input_one==5) and """
                every input in inputs contains:
                    observation: np.ndarray
                    action: np.ndarray or int or float
                    reward: float
                    next_observation: np.ndarray
                    info: dict
                    network_onnx_path: str
                    tensorboard_dir: str
            """
            assert isinstance(input_one[0],np.ndarray) and 'InputError: observation should be np.ndarray type'
            assert isinstance(input_one[1],(np.ndarray,int,float)) and 'InputError: action should be np.ndarray or int or float type'
            assert isinstance(input_one[2], float) and 'InputError: reward should be float type'
            assert isinstance(input_one[3], np.ndarray) and 'InputError: observation should be np.ndarray type'
            assert isinstance(input_one[4],dict) and 'InputError: info should be dict type'

