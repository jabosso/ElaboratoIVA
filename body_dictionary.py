class body():
    def __init__(self):
        # self.dictionary={
        #     "head" : 0,
        #     "neck" : 1,
        #     "dx_shoulder": 2,
        #     "dx_elbow" : 3,
        #     "dx_hand" : 4,
        #     "sx_shoulder": 5,
        #     "sx_elbow": 6,
        #     "sx_hand": 7,
        #     "pelvis" : 8,
        #     "dx_hip" : 9,
        #     "dx_knee": 10,
        #     "dx_foot": 11,
        #     "sx_hip": 12,
        #     "sx_knee": 13,
        #     "sx_foot": 14
        # }
        self.dictionary={
            0:"head",
            1:"neck",
            2:"dx_shoulder",
            3: "dx_elbow",
            4:"dx_hand",
            5:"sx_shoulder",
            6:"sx_elbow",
            7:"sx_hand",
            8:"pelvis",
            9:"dx_hip",
            10:"dx_knee",
            11:"dx_foot",
            12:"sx_hip",
            13:"sx_knee",
            14:"sx_foot"
        }
        self.connection =  [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
              (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)]

