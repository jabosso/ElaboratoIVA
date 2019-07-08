import argparse
from video import video_to_matrix
from linear_transformation import *
from utility_tools import *
from csv_tools import *
import body_dictionary as body_dic
from time_tools import *
from dtw import dtw
# from second_main import carico_vettore, distance_cos
import time

flag_sampling = False

body = body_dic.body()


type_exercise = ['ArmsWarmUp', 'prova']

ap = argparse.ArgumentParser()
ex = 0

ap.add_argument("-e", "--exercise", required=True, help='Chose exercise')
ap.add_argument("-v", "--video", required=True, help="Video")  # user video
args = vars(ap.parse_args())
print("{}".format(args["exercise"]))
ex = args["exercise"]  # ex= 'Arms2'
vid_na = args['video']
interest_path = 'move/models/' + ex + '/interest_point.txt'

model_path = 'move/models/' + ex + '/cycle/model.csv'

model = csv_to_matrix(model_path)
model['frame'] = model['frame'].astype((int))
model['x'] = model['x'].astype(int)
model['y'] = model['y'].astype(int)
interest_index = get_interest_point(interest_path)

matrix, total, _ = video_to_matrix(vid_na)
print(matrix.shape)
if matrix[0][1][1] is None or matrix[0][5][1] is None:
    rot_ref =(8,12)
else:
    rot_ref = (1,5)
user_matrix = linear_transformation(matrix,rot_ref)
data = create_dataframe(user_matrix, ['frame', 'x', 'y', 'score'])

data = add_body_parts(data, body.dictionary)
data = remove_not_interest_point(interest_path, data)
data['frame'] = data['frame'].astype(int)
data['x'] = data['x'].astype(int)
data['y'] = data['y'].astype(int)
mid_points = cycle_identify(data)
lista_w = list_cycle(total, mid_points)
data_c = generate_cycle_model(data, mid_points)
n_body = len(pd.unique(data_c[0].body_part))
for i in range(len(data_c)):
    data_c[i] = shifted(data_c[i], n_body)
    data_c[i] = normalize_dataFrame(data_c[i])
    # -----sampling-------
    if flag_sampling:
        data_c[i] = sampling(data_c[i])
        data_c[i] = shifted(data_c[i], n_body)

    # -----------

shape1 = len(pd.unique(model['frame']))
shape2 = len(data_c)
list_p = np.full((shape1, shape2), np.inf)

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 1
fontColor = (255, 0, 255)
# lineType               = 2

for i in range(shape2):
    path = dtw(model, data_c[i])
    p = sincro(path)
    for element in p:
        j = element[0]
        value_ = element[1]
        list_p[j][i] = value_
dim_v = len(pd.unique(model['body_part']))

variance_w = csv_to_matrix('move/models/' + ex + '/cycle/variance.csv')
medium_cs = csv_to_matrix('move/models/' + ex + '/cycle/medium.csv')
color1 = [0, 0, 255]
color2 = [0, 255, 0]

for i in range(shape2):
    total_score = 0
    vect = []
    current_cycle = data_c[i]
    current_s = lista_w[i]
    worse_poses = [(0, 0, 0)]
    if flag_sampling:
        current_s = sampling_mat(current_s)
    for j in range(shape1):
        if list_p[j][i] != np.inf:
            vect.append([j, list_p[j][i]])
    for elem in vect:
        f_u = current_cycle.loc[current_cycle['frame'] == elem[1]]
        v_u = vector_load(f_u, dim_v)
        distance_user = distance_cosine(v_u, dim_v, interest_index)
        f_m = model.loc[model['frame'] == elem[0]]
        v_m = vector_load(f_m, dim_v)
        distance_model = distance_cosine(v_m, dim_v, interest_index)
        current_variance = variance_w.loc[variance_w['frame'] == elem[0]]

        current_medium = medium_cs.loc[medium_cs['frame'] == elem[0]]
        varia = current_variance['variance']

        medium_dist = current_medium['medium_dist']
        result = np.zeros(len(varia))
        check = np.zeros(len(varia))
        pose_score = 0
        for i in range(len(varia)):
            result[i] = math.fabs(distance_user[i] - medium_dist.iloc[i])

            if (round(result[i], 3) < round(varia.iloc[i], 3)) or (round(result[i], 3) == round(varia.iloc[i], 3)):
                check[i] = 1
            else:
                pose_score = pose_score + (round(result[i], 3) - round(varia.iloc[i], 3))

        if pose_score > worse_poses[0][1] or pose_score == 0:
            if pose_score == 0:
                worse_poses[0] = (elem, pose_score, check)

            elif len(worse_poses) < 8:
                worse_poses.append([elem, pose_score, check])
            else:
                worse_poses[0] = (elem, pose_score, check)

            worse_poses.sort(key=SecondSort)
        for element in check:
            total_score = total_score + element
    total_score = total_score / (len(varia) * len(vect))
    visual_worse(worse_poses, current_cycle, model, current_s, total_score)
