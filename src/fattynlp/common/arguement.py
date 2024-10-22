import json

class Arguements:
    def __init__(self):
        pass

    @classmethod
    def create_args(cls):
        return cls()

    @classmethod
    def from_json(cls, json_fpath):
        args = cls.create_args()
        with open(json_fpath) as f:
            dic = json.loads(f.read())
            args.set_param(dic)
        return args

    def _to_dict(self, d):
        dic = self.__dict__
        for k, v in dic.items():
            if isinstance(v, Arguements):
                tmp_d = {}
                v._to_dict(tmp_d)
                d[k] = tmp_d
            else:
                d[k] = v
    
    def to_dict(self):
        d = dict()
        self._to_dict(d)
        return d

    def __str__(self):
        d = self.to_dict()
        return json.dumps(d, ensure_ascii=False, indent=4)

    def set_param(self, dic):
        for k, v in dic.items():
            assert isinstance(k, str)
            if isinstance(v, dict):
                sub_args = self.create_args()
                sub_args.set_param(v)
                setattr(self, k, sub_args)
            else:
                setattr(self, k, v)

