#%%
import torch
import os
os.environ['DATA_GRAND_CENTRAL'] = '/data5/eddie/calvin'
task_to_validation = torch.load(f'{os.environ["DATA_GRAND_CENTRAL"]}/task_to_validation_all.pt', map_location="cpu")

# %%
# generate eval_sequences

eval_sequences = []
for t in task_to_validation:
    if t == 'unstack_block':
        continue
    (
        rel_actions,
        robot_obs,
        scene_obs,
        depth_gripper,
        rgb_static,
        rgb_gripper,
        depth_gripper_resized,
        ann,
        tasks,
        embeddings,
        goal_image,
    ) = task_to_validation[t]
    try:
        # for i in range(int(eval_task_dist[t] / 1000 * 115)):
        for i in range(len(robot_obs)):
            eval_sequences.append([(robot_obs[i][0], scene_obs[i][0], goal_image[i]), (t,)])
    except IndexError:
        continue
    except KeyError:
        continue
    
torch.save(eval_sequences, 'eval_sequences.pt')

# %%

eval_task_dist = {
    "open_drawer": 115,
    "move_slider_right": 87,
    "move_slider_left": 64,
    "turn_on_led": 56,
    "turn_off_led": 54,
    "turn_on_lightbulb": 41,
    "close_drawer": 41,
    "push_blue_block_left": 35,
    "push_red_block_left": 35,
    "turn_off_lightbulb": 34,
    "rotate_pink_block_right": 31,
    "push_pink_block_left": 31,
    "lift_blue_block_slider": 30,
    "push_pink_block_right": 29,
    "push_red_block_right": 29,
    "rotate_red_block_right": 29,
    "rotate_blue_block_right": 28,
    "lift_red_block_slider": 26,
    "lift_pink_block_slider": 25,
    "lift_pink_block_table": 25,
    "rotate_blue_block_left": 24,
    "lift_red_block_table": 24,
    "lift_blue_block_table": 24,
    "push_blue_block_right": 23,
    "rotate_pink_block_left": 23,
    "rotate_red_block_left": 22,
    "push_into_drawer": 15,
}
def reweight(d):
    score = 0
    for i in eval_task_dist:
        score += eval_task_dist[i] / 1000 * d[i]
    j# print('Reweighted score:', score)
    return score
