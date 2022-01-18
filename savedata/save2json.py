import random
import numpy as np
from datetime import datetime
import os

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
        

class save2json(object):
    def __init__(self,
                version_id,
                capacity:int,
                model_type='onnx',
                check_input=True,
                check_model_pth=True,
                check_tensorboard_pth=True) -> None:
        super(self,save2json).__init__()


        self.capacity=capacity
        self.model_type=model_type
        self.check_input=check_input
        self.check_model_pth=check_model_pth
        self.check_tensorboard_pth=check_tensorboard_pth


        """
            {
                VERSION:{
                    InputData:ReplayBuffer,
                    ModelPath:{
                        TIME1:ModelPath1,
                        TIME2:ModelPath2,
                        ...
                    }
                    TensorboardPath:{
                        TIME1:TensorboardPath1,
                        TIME2:TensorboardPath2,
                        ...
                    }
                },
                METAINFO:{
                    VERSION:{
                        Model Type:Model Type,
                        Replay Buffer Size:capacity
                    }
                }
            }
        """

        self.js={}
        self.js['metaInfo']={}
        self.version_name=self._get_version_format(version_id)
        self.js['metaInfo'][self.version_name]={
            "Model Type":model_type,
            "Replay Buffer Size":capacity
        }
        if self.version_name not in self.js.keys():
            self.js[self.version_name]={
                "InputData":ReplayBuffer(capacity=capacity),
                "ModelPath":{},
                "TensorboardPath":{}
            }

    def __len__(self,version_id=None):
        version_name=self.version_name if version_id==None else self._get_version_format(version_id)

        return self.js['metaInfo'][version_name]["Replay Buffer Size"]
    def save2json(self):
        pass
    def loadjson(self,path):
        pass

            
    def push(self,inputs):
        for i,input_one in enumerate(inputs):
            check=None
            if i!=1:
                check=False
            self.push_single(input_one,check)

    def push_single(self,input_one,check=None):
        if check==None:
            check=self.check_input

        if check:
            self._check_input_data(input_one)

        self.js[self.version_name]["InputData"].push(*input_one)
    
    def update_model_path(self,model_pth):
        NOW=datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.js[self.version_name]['ModelPath'][NOW]=model_pth

    def update_tensorboard_path(self,tensorboard_pth):
        """
            FORMAT: ("%Y_%m_%d_%H_%M")
        """
        NOW=datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.js[self.version_name]['TensorboardPath'][NOW]=tensorboard_pth

    def get_replay_buffer(self,batch_size,version_id=None):
        version_name=self.version_name if version_id==None else self._get_version_format(version_id)
        
        assert version_name in self.js.keys()
        state, action, reward, next_state, done=self.js[version_name]["InputData"].sample(batch_size)
        return state, action, reward, next_state, done

    def get_model_path(self,version_id=None):
        version_name=self.version_name if version_id==None else self._get_version_format(version_id)

        return self.js[version_name]["ModelPath"]

    def get_tensorboar_path(self,version_id=None):
        version_name=self.version_name if version_id==None else self._get_version_format(version_id)

        return self.js[version_name]["TensorboardPath"]

    def get_latest_model_pth(self,version_id=None):
        version_name=self.version_name if version_id==None else self._get_version_format(version_id)

        Keys=[datetime.strptime("1999-01-11", "%Y_%m_%d_%H_%M") for k in self.js[version_name]["ModelPath"].keys() ]
        Keys=sorted(Keys)
        return self.js[version_name]["ModelPath"][Keys[-1].strftime("%Y_%m_%d_%H_%M")]

    def get_latest_tensorboard_pth(self,version_id=None):
        version_name=self.version_name if version_id==None else self._get_version_format(version_id)

        Keys=[datetime.strptime("1999-01-11", "%Y_%m_%d_%H_%M") for k in self.js[version_name]["TensorboardPath"].keys() ]
        Keys=sorted(Keys)
        return self.js[version_name]["TensorboardPath"][Keys[-1].strftime("%Y_%m_%d_%H_%M")]

    def delete_certain_version(self,version_id=None):
        version_name=self.version_name if version_id==None else self._get_version_format(version_id)

        self.js[version_name]={}
        self.js['metaInfo'][version_name]={}
        
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



