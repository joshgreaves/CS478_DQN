
import numpy as np
class DQN(object):
    
    def __init__(self, topology, discount=0.95, current_update_countdown=1000, p_random=):
		#somehow make the topology into the current_net and the learning_net

		self.discount = discount
		self.current_update_countdown = current_update_countdown
		self.p_random = p_random

	def train_step(self, s, a, r, s_prime)
		learning_net_q = self.learning_net.forward(s)
		pred = learning_net_q[:,a]

		current_net_q = self.current_net.forward(s_prime)
		target = r + self.discount * np.max(current_net_q, axis=1)

		loss_vector = target - pred
		loss = np.sum(np.power(np.clip(loss_vector, -1, 1), 2))

		# learning_net.backwards(loss) Something like this.. Whatever is normal for Tensorflow in backprop

		self.current_update_countdown -= 1
		if self.current_update_countdown <= 0:
			# replace wieghts of current_net with weights of learning_net

	# return index of selected action
	def select_action(self, q_values, p_random=self.p_random):
		if np.random.rand() > p_random:
			return argmax(q_values)
		else:
			return np.random.randint(0, q_values.shape[0])
	
			
	def episode_step(self, state):
		return self.select_action(self.current_net.forward(state))

	def set_p_random(self, p_random):
		self.p_random = p_random
