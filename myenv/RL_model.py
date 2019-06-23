from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from myenv.rl_othello import OthelloDQNAgent

class DQNmodel:
    def __init__(self,input_size=(8,8,2),output_size=65):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + input_size))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(output_size))
        model.add(Activation('linear'))

        memory = SequentialMemory(limit=50000, window_length=1)

        self.dqn = OthelloDQNAgent(model=model, nb_actions=output_size, memory=memory, nb_steps_warmup=10,target_model_update=1e-2)
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])
