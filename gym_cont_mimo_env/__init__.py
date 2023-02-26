from gym.envs.registration import register

register(
    id='mimo-v0',
    entry_point='gym_cont_mimo_env.envs:MIMOEnv',
)

register(
    id='mimo-v1',
    entry_point='gym_cont_mimo_env.envs:MIMOEnv_v1',
)

register(
    id='mimo-v2',
    entry_point='gym_cont_mimo_env.envs:MIMOEnv_v2',
)

register(
    id='mimo-v3',
    entry_point='gym_cont_mimo_env.envs:MIMOEnv_v3',
)

register(
    id='mimo-v4',
    entry_point='gym_cont_mimo_env.envs:MIMOEnv_v4',
)

register(
    id='mimo-v5',
    entry_point='gym_cont_mimo_env.envs:MIMOEnv_v5',
)

register(
    id='mimo-v6',
    entry_point='gym_cont_mimo_env.envs:MIMOEnv_v6',
)
