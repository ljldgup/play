import os

data_dir = os.getcwd()
env_colomns = ['role1_action', 'role2_action',
               'role1_energy', 'role2_energy',
               'role1_position_x', 'role1_position_y',
               'role2_position_x', 'role2_position_y', 'role1_baoqi', 'role2_baoqi', 'role1', 'role2',
               'role1_guard_value',
               'role1_combo_count',
               'role1_life', 'role2_life',
               'time', 'coins', ]
# 进入函数raw_env_data_to_input的列
input_colomns = ['role1_action', 'role2_action',
                 'role1_energy', 'role2_energy',
                 'role1_position_x', 'role1_position_y',
                 'role2_position_x', 'role2_position_y', 'role1_baoqi', 'role2_baoqi', 'action']