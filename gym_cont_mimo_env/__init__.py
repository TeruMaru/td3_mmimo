from gym.envs.registration import register

register(
    id='mimo-v0',
    entry_point='gym_cont_mimo_env.envs:MIMOEnv',
)