import gym
from gym import spaces
import numpy as np
from scipy.linalg import toeplitz, sqrtm


class MIMOEnv(gym.Env):
    def __init__(self, L, K, M, ASD_deg, square_length=1000, accuracy=2,
                max_power_per_UE=100, nbr_realizations=100,
                delta_rho=5, debug=False, val=False):
        '''
        L = Number of BSs and cells
        K = Number of UEs per cell
        M = Number of antennas per BS

        square_length = Length of an edege of the square network area in meters, default to 1km taken from the book
                (4 cells per dim, 0.25km per cell).

        ASD_deg = scalar Angular standard deviation around the nominal angle
                   (measured in degrees)
        accuracy = Compute exact correlation matrices from the local scattering
                model if approx = 1. Compute a small-angle approximation of the
                model if approx = 2
        '''
        self.M = M
        self.K = K
        self.L = L
        self.ASD_deg = ASD_deg
        self.accuracy = accuracy

        self.no_cell_per_dim = int(np.sqrt(L))
        self.square_length = square_length
        self.nbr_realizations = nbr_realizations
        # Maximum power a BS can allocate to a UE interms of mW
        self.max_power_per_UE = float(max_power_per_UE)
        self.delta_rho = delta_rho  # default to 5mW
        self.debug = debug

        self.max_steps = 1000
        self.max_keep = 30
        self.validate = val

        # Deploy BSs and its wrapped topology
        cell_length, _, BSs_pos_flatten, BSs_pos_wrapped = self.deploy_BSs()
        self.cell_length = cell_length
        self.BSs_pos_flatten = BSs_pos_flatten
        self.BSs_pos_wrapped = BSs_pos_wrapped

        # gym required variables
        '''
         Initialize the action space, which is a tuple of
           + The agent ID, i.e. the considering BS
           + The UE affected by lowest downlink spectral efficiency
           + Delta power allocated to that BS
        '''
        # self.action_space = spaces.Tuple(tuple(
        #     [space.Box(-self.delta_rho ,self.delta_rho)]
        # )
        # )
        self.action_space = spaces.Box(low=-self.delta_rho,
                                        high=self.delta_rho,
                                        shape=(self.K*self.L,),
                                        dtype=np.float64)



    def _get_obs(self):
        # Normalize distance by changing unit from meters to kilometers -> range [0,1]
        dist_mask = np.kron(self.UEs_BSs_dist / 1e3, [1, 0, 0])
        # Normalize angle by dividing to pi -> range [-1,1]
        angle_mask = np.kron(self.UEs_BSs_angle / np.pi, [0, 1, 0])
        # Normalize power by diving to max power -> range [0,1]
        power_mask = np.kron(self.rho / self.max_power_per_UE, [0, 0, 1])

        obs = (dist_mask + angle_mask + power_mask).flatten()
        return obs

    def reset(self, ref_data=None):
        self.elapsed_step = 0
        self.consecutive_keep = 0
        # K by L real distance from UEs to BS in current episode
        if self.validate:
            assert ref_data.shape == (
                self.K, self.L), "Reference data is not of shape KxL"
            UEs_pos = ref_data
        else:
            UEs_pos = self.populate_UEs()

        # K by L real down link power allocated to kth UE by jth BS
        # self.rho = np.ones((self.K, self.L)) * self.max_power_per_UE * 0.5
        rand_rho = np.random.rand(self.K, self.L)
        norm_rho = rand_rho/rand_rho.sum(axis=0, keepdims=1)
        self.rho = norm_rho * self.K * self.max_power_per_UE
        
        signal_MMMSE, interf_MMMSE, prelog_factor = self.get_episode_SINR_components(UEs_pos)

        self.avg_channel_gains = signal_MMMSE
        self.avg_interference_gains = interf_MMMSE

        self.prelog_factor = prelog_factor

        self.min_SE = np.amin(self.compute_DL_SE(self.rho,
                                                 self.avg_channel_gains,
                                                 self.avg_interference_gains,
                                                 self.prelog_factor))

        self.max_min_SE = self.min_SE
        return self._get_obs()

    def step(self, action):
        assert self.action_space.contains(action), ("Invalid action provided."
        f" An action is a 1D array of shape ({self.K}, {self.L})"
        f" where each element range from ({-self.delta_rho}, {self.delta_rho})")

        action = action.reshape((self.K, self.L))
        self.rho += action

        # If no BS has its allocated power exceeds maximum allowable value,
        # set done to Fasle and compute reward
        if np.all(np.sum(self.rho, axis=0) <= self.K * self.max_power_per_UE) \
        and np.all(self.rho >= 0):
            if np.all(self.rho == 0):
                done = True
                reward = -10
            else:
                reward, done = self.compute_reward()
        else:
            done = True
            reward = -10

        return self._get_obs(), reward, done, {}

    def compute_reward(self):
        # Compute spectral efficiency
        # Compute spectral efficiency
        SE = self.compute_DL_SE(self.rho, self.avg_channel_gains,
                                self.avg_interference_gains,
                                self.prelog_factor)
        min_SE = np.amin(SE)
        done = False
        # Compare current overall SE with previous overall SE,
        # if it is better the reward, else not
        if min_SE > self.max_min_SE:
            self.max_min_SE = min_SE
            if self.consecutive_keep > 0:
                self.consecutive_keep = 0
            reward = 1
        elif min_SE == self.max_min_SE:
            self.consecutive_keep += 1
            # check if SE is saturated
            if self.consecutive_keep >= self.max_keep:
                # Encourage the agent to keep SE constant
                # if it can not be improved anymore
                reward = 10
                done = True
            else:
                # Small punishment for keeping power allocation scheme unchanged
                reward = -1
        else:
            reward = -10
            done = True
        self.min_SE = min_SE
        return reward, done

    def get_episode_SINR_components(self, UEs_pos):
        # Communication bandwidth
        B = 20e6

        # Total uplink transmit power per UE (mW)
        p = 100

        # Pilot reuse factore
        reuse_factor = 1

        # Total transmission sequence length
        tau_c = 200

        # Generates the channel statistics between UEs at random
        # locations and the BSs in the running example, defined in Section 4.1.3.
        R_unscaled, channel_gain_dB = self.compute_correlation_matrix(self.BSs_pos_wrapped, UEs_pos)

        # Define noise figure at BS (in dB)
        noise_figure = 7

        # Compute noise power
        noise_variance_dBm = -174 + 10*np.log10(B) + noise_figure

        # Compute channel gain over noise
        channel_gain_over_noise = channel_gain_dB - noise_variance_dBm

        # Sample CSI's real components from normal distribution with mean = 0 , std=1
        H_real = np.random.standard_normal(
            size=(self.M, self.nbr_realizations, self.K, self.L, self.L))
        # Sample CSI's imaginary components from normal distribution with mean = 0 , std=1
        H_imag = np.random.standard_normal(
            size=(self.M, self.nbr_realizations, self.K, self.L, self.L))
        H = H_real + (1j*H_imag)

        R_scaled, H_scaled = self.create_channel_side_info(R_unscaled, H, channel_gain_over_noise)

        tau_p, H_hat, C = self.uplink_channel_estimation(R_scaled, H_scaled, p, reuse_factor)

        signal_MMMSE, interf_MMMSE, prelog_factor = self.get_DL_SINR_comps(H_scaled, H_hat, C, tau_c, tau_p, p)

        return signal_MMMSE, interf_MMMSE, prelog_factor

    def deploy_BSs(self):
        '''
        TODO: Write description
        '''
        # Number of BS per cell
        no_BS_per_dim = self.no_cell_per_dim

        # Distance between BSs in vertical/horizontal direction, can also be depicted as cell's length
        inter_BS_dist = self.square_length / no_BS_per_dim
        cell_length = inter_BS_dist

        #Deploy BSs on the grid at the center of each cell
        cell_center_offset = cell_length / 2
        ## Create an array of horizontal coordinates x of the BSs, which are the x cooridnates
        ## of the centers of each cell
        BSs_x_pos = np.arange(cell_center_offset,
                              self.square_length, inter_BS_dist)
        ## Create vertical coordinates y of the BSs
        # The axis is to make sure broadcasting works
        BSs_y_pos = np.expand_dims(BSs_x_pos, axis=1)
        #BSs_xy_pos = BSs_x_pos + 1j*BSs_y_pos
        BSs_xy_pos = BSs_y_pos + 1j*BSs_x_pos
        BSs_xy_pos_flatten = BSs_xy_pos.reshape((self.L,))
        ## Expand to create wrap-around topology
        BSs_wrapped_pos = np.tile(BSs_xy_pos, (3, 3))
        ## Create a 3x3 mask for the wrapped around cells
        mask_x = np.arange(-self.square_length, 2 *
                           self.square_length, self.square_length)
        mask_y = np.expand_dims(mask_x, axis=1)
        mask = mask_x + 1j*mask_y
        ## Each member of the mask will be broadcast-summed into the original position matrix
        ## to create a wrapped around position matrix
        # for i in range(3):
        #     for j in range(3):
        #         BSs_wrapped_pos[no_BS_per_dim*i:no_BS_per_dim*(i+1), no_BS_per_dim*j:no_BS_per_dim*(j+1)] += mask[i, j]
        BSs_wrapped_pos += np.kron(mask,
                                   np.ones((no_BS_per_dim, no_BS_per_dim)))
        return cell_length, BSs_xy_pos, BSs_xy_pos_flatten, BSs_wrapped_pos

    def populate_UEs(self):
        '''
        TODO: Write description
        '''
        # Minimum distance between BSs and UEs
        min_dist = 35
        # Prepare to put out UEs in the cells
        UEs_pos = np.zeros((self.K, self.L), dtype=complex)
        UEs_per_cell = np.zeros(self.L, dtype=int)

        # Go through all cells
        for l in range(self.L):
            # Place K UEs in the cell, uniformly at random. The procedure is
            # iterative since UEs that do not satisfy the minimum distance are
            # replaced with new UEs
            while UEs_per_cell[l] < self.K:
                # Place new UE in the cell
                remaining_UE = self.K - UEs_per_cell[l]
                batch_pos_X = np.random.rand(
                    remaining_UE) * self.cell_length - self.cell_length/2
                batch_pos_Y = np.random.rand(
                    remaining_UE) * self.cell_length - self.cell_length/2
                batch_pos_XY = batch_pos_X + 1j*batch_pos_Y

                sel_posXY = batch_pos_XY[np.abs(batch_pos_XY) >= min_dist]
                UEs_pos[UEs_per_cell[l]:UEs_per_cell[l] + sel_posXY.shape[0],
                        l] = sel_posXY + self.BSs_pos_flatten[l]
                UEs_per_cell[l] += sel_posXY.shape[0]
        return UEs_pos

    def compute_correlation_matrix(self, BSs_wrapped_pos, UEs_pos):
        '''
        TODO: Write description
        '''
        # pathloss expoenent:
        alpha = 3.76

        # Median channel gain in dB at a reference distance of 1 kilometer
        upsilon = -148

        # Median channel gain in dB at a reference distance of 1 meter
        med_gain_per_meter = upsilon - 10 * alpha * np.log10(1e-3)

        # Define the antenna spacing (in number of wavelengths)
        antenna_spacing = 1/2    # Half-wavelength distance

        # Prepare to store normalized spatial correlation matrices
        R = np.zeros((self.M, self.M, self.K, self.L, self.L), dtype=complex)

        # Prepare to store average channel gain numbers (in dB). This is \Beta in equation (2.3)
        channel_gain_dB = np.zeros((self.K, self.L, self.L))

        # Prepare to store distance from K UEs to L BSs
        kUEs_jBS_dist = np.zeros((self.K, self.L))

        # Prepare to store angle from K UEs to L BSs
        kUEs_jBS_angle = np.zeros((self.K, self.L))

        # Go through all cells
        for l in range(self.L):
            # Having the positions of K users in cell l, we go on to compute the distance
            # between these users with all BSs, where the wrap around topology is handy
            # to make sure the distances are as short as possible. For example, the distance
            # between K users in cell l=0 to BS j=3 in the original network will transform to
            # the distance between K users in cell l=0 in the orignal network with the BS j=3
            # in the adjacent wrap-around networks to its left (see visualization in jupyter notebook)
            ## Loop through all BSs
            for j in range(self.L):
                ## Get positions of jth BSs in all wrapped cells
                j_doppelgangers = self.get_wrapped_BSj(BSs_wrapped_pos, j)
                # reshape to (9,1) to make sure broadcasting works
                j_doppelgangers = np.expand_dims(j_doppelgangers, axis=1)
                ## Distance between K users in cell lth and jth BSs, result of a broadcasted substraction
                ## between (K,) ndarray and (9,1) ndarray, should be of shape (9,K)
                lj_diff = UEs_pos[:, l] - j_doppelgangers
                if self.debug:
                    assert lj_diff.shape == (
                        9, self.K), "lj_diff is not of shape (9,K)"
                lj_dist = np.abs(lj_diff)
                lj_angle = np.angle(lj_diff)
                ## Select only min distances, so the result should be of shape (K,)
                # Coresponds to whichpos variable in matlab code
                sel_rows = np.argmin(lj_dist, axis=0)
                # Coresponds to distancesBSj in matlab code
                kUEs_jBS_dist[:, j] = lj_dist[sel_rows, np.arange(self.K)]

                # Compute average channel gain using the large-scale fading model in
                # (2.3), while neglecting the shadow fading part (F_{lk}^j)
                channel_gain_dB[:, l, j] = med_gain_per_meter - \
                    alpha * 10 * np.log10(kUEs_jBS_dist[:, j])

                # Compute nominal angles between UE k in cell l and BS j, and
                # generate spatial correlation matrices for the channels using the
                # local scattering model
                for k in range(self.K):
                    # Angle from kth UE in lth cell to nearest jth BS
                    kUEs_jBS_angle[k, j] = lj_angle[sel_rows[k], k]

                    if self.accuracy == 1:
                        #Use the exact implementation of the local scattering model
                        pass
                    elif self.accuracy == 2:
                        #Use the approximate implementation of the local scattering model
                        R[:, :, k, l, j] = self.local_scattering_R_approx(
                            kUEs_jBS_angle[k, j], antenna_spacing)
                        pass
                # Not considering shadowing
        return R, channel_gain_dB

    def get_wrapped_BSj(self, wrapped_BSs_mat, jth_flatten):
        '''
        Given the wrapped-around position matrix of all BSs in the network and the jth BS being considered,
        deduce where jth BS is in the remaining networks (the wrapped around ones)
        INPUT:
            wrapped_BSs_mat = the (3 * sqrt(L), 3 * sqrt(L)) matrix of wrapped around BSs' positions
            jth_flatten = the index of the jth BS being considered in the flatten version of the orignal network
        OUTPUT:
            An ndarray of shape (9,) containing the 9 locations of jth BS
        '''
        assert wrapped_BSs_mat.shape == (3*self.no_cell_per_dim, 3*self.no_cell_per_dim), "Wrap position matrix is not in the desired shape. For L={l}, wrap position matrix is expected to be a square matrix of shape ({row},{row})".format(
            l=self.no_cell_per_dim ** 2, row=3*self.no_cell_per_dim)
        # Conver jth_flatten to 2D indices in the orignal network matrix
        jth_row = jth_flatten // self.no_cell_per_dim
        jth_col = jth_flatten % self.no_cell_per_dim
        # Since we are using a 3x3 wrap around network, our orignal network would be at the center.
        # Therefore, jth_row/col in the orignal network would be (jth_row/col + 4) in wrapped_BSs_mat
        # Using this same deduction, we generallize to find all 9 locations of jth BS in wrapped_BSs_mat
        wrapped_pos = np.zeros((9,), dtype=complex)
        for index in range(9):
            if index == 0:
                wrapped_pos[index] = wrapped_BSs_mat[jth_row, jth_col]
            else:
                wrapped_row = jth_row + ((index // 3) * self.no_cell_per_dim)
                wrapped_col = jth_col + ((index % 3) * self.no_cell_per_dim)
                wrapped_pos[index] = wrapped_BSs_mat[wrapped_row, wrapped_col]

        return wrapped_pos

    def local_scattering_R_approx(self, nominal_angle, antenna_spacing):
        '''
        Generate the spatial correlation matrix for the local scattering model,
        defined in (2.23) with the Gaussian angular distribution. The small-angle
        approximation described in Section 2.6.2 is used to increase efficiency,
        thus this function should only be used for ASDs below 15 degrees.
        Note: The ouput MxM R matrix is not yet multiplied with the \Beta eq (2.23)
        '''
        # Compute the ASD in radians based on input
        ASD_rad = self.ASD_deg * np.pi / 180
        # The correlation matrix has a Toeplitz structure, so we only need to
        # compute the first column of the matrix
        d_h = antenna_spacing
        phi_bar = nominal_angle
        # The - is for compatability with Matlab's orignal code, but I think there
        # is a bug there. (2.24) says R_{l,m} = ...(l-m)...(l-m), so if they cal
        # 1st row, m is varrying, then their distance variable should be negative
        dist_from_first_atn = - np.arange(self.M)

        # Follow Equation (2.24)
        first_col = np.exp(1j*2*np.pi*d_h*dist_from_first_atn*np.sin(phi_bar)) * \
            np.exp(-((ASD_rad**2)/2) *
                   ((2*np.pi*d_h*dist_from_first_atn*np.cos(phi_bar))**2))

        # Create toeplitz matrix from first column
        R = toeplitz(first_col)
        return R

    def create_channel_side_info(self, R_incomplete, standard_H, channel_gain_dB):
        '''
        Generate CSI matrix H from correlation matrix R using Normal distribution
        INPUT:
            R_incomplete      : a M x M x K x L x L tensor of M x M toeplitz correlation matrices
                                for channel realizations between kth user in lth cell and jth BS (each BS has M
                                antenas). R follows eq (2.24) in Emil Bjornson's Massive MIMO monograph but
                                have yet been scaled with \Beta, so it is tagged with incomplete
            standard_H        : a M x N x K x L x L tensor of channels' responses drawn form complex Gaussian
                                distribution with mean=0, std=1
            channel_gain_dB   : K x L x L average channel gain in dB tensor whose elements are
                                the normalized trace of the complete R[:,:,k,l,j] correlation matrices for k in K,
                                l in L and j in L
        OUTPUT:
            R: tensor of shape M x M x K x L x L of the finalized correlation matrix
            H: tensor of shape M x nbr_sample x K x L x L (tried to keep the same dimensional
            as compared to the reference code)
        '''
        # Create a placeholder for beta and final R
        beta = np.zeros((self.K, self.L, self.L))
        R = np.zeros_like(R_incomplete)
        # Sample real components from normal distribution with mean = 0 , std=1
        # H_real = np.random.standard_normal(size=(self.M, nbr_sample, self.K, self.L, self.L))
        # # Sample imaginary components from normal distribution with mean = 0 , std=1
        # H_imag = np.random.standard_normal(size=(self.M, nbr_sample, self.K, self.L, self.L))

        # H = H_real + (1j*H_imag)
        nbr_sample = standard_H.shape[1]
        H = standard_H.copy()
        # Scale H from std = 1 to the desired std
        for l in range(self.L):
            for j in range(self.L):
                for k in range(self.K):
                    jkl_gain = channel_gain_dB[k, l, j]
                    if jkl_gain > (-np.inf):
                        # Compute beta from its dB correspondence. Should be a scalar
                        beta[k, l, j] = 10 ** (jkl_gain/10)
                        if self.debug:
                            assert isinstance(
                                beta[k, l, j], float), "jkl_beta is not a scalar!"
                        # Finalize R for channel jkl
                        R[:, :, k, l, j] = beta[k, l, j] * \
                            R_incomplete[:, :, k, l, j]
                        if self.debug:
                            assert R[:, :, k, l, j].shape == (
                                self.M, self.M), "jkl_R is not a MxM matrix"
                        # Extract M x M standard deviation matrix for k,l,j channel
                        jkl_std = sqrtm(R[:, :, k, l, j])
                        if self.debug:
                            assert jkl_std.shape == (
                                self.M, self.M), "jkl_std is not a MxM matrix"
                        # Scale H_{k,l,j} from std = 1 to std = jkl_std. The sqrt(0.5) follows
                        # complex normal distribution rule
                        H[:, :, k, l, j] = np.sqrt(
                            0.5) * (jkl_std.dot(H[:, :, k, l, j]))
                    else:
                        # If the gain is neglectable
                        R[:, :, k, l, j] = np.zeros(
                            (self.M, self.M), dtype=complex)
                        H[:, :, k, l, j] = np.zeros(
                            (self.M, nbr_sample), dtype=complex)  # No channel
        return R, H

    def uplink_channel_estimation(self, R, H, ul_power, pilot_reuse_factor, N=None):
        '''
            Estimate the channel in UL (up link, UEs to BSs direction) using knowledge of convariance matrix R
            and realizations of H
            INPUT:
                R: correlation matrix of shape M x M x K x L x L characterizing local scattering model
                H: channel realizations of shape M x nbrOfRealizations x K x L x L
                ul_power: UL transmit power per UE, same for every UEs
                pilot_reuse_factor: pilot reuse factor. Higher reuse factor results in lower intercell
                        inteference during UL channel estimation, but increase pilot length
            OUTPUT:
                H_hat: channel estimation of the same shape as H, derived from MMSE algorithm
                C: M x M x K x L x L tensor of estimation error correlation matrices
        '''
        if self.debug:
            assert R.shape == (self.M, self.M, self.K, self.L,
                               self.L), "Correlation tensor is not with shape M x M x K x L x L"
            assert H.shape == (self.M, self.nbr_realizations, self.K, self.L,
                               self.L), "CSI tensor is not with shape M x Realizations x K x L x L"

        # Length of pilot sequences
        tau_p = pilot_reuse_factor * self.K

        # Generate pilot pattern. Refer to figure 4.4 in the monograph, as the generated
        # patern is the flatten version of the color coded matrices.
        if pilot_reuse_factor == 1:
            pilot_ptrn = np.ones(self.L)
        elif pilot_reuse_factor == 2:
            pilot_ptrn = np.kron(
                np.ones(2,), np.array([1, 2, 1, 2, 2, 1, 2, 1]))
        elif pilot_reuse_factor == 4:
            pilot_ptrn = np.kron(
                np.ones(2,), np.array([1, 2, 1, 2, 3, 4, 3, 4]))
        elif pilot_reuse_factor == 16:
            pilot_ptrn = np.arange(self.L)

        # Store identity matrix of size M x M
        eye_M = np.eye(self.M)

        # Generate realizations of normalized noise if not inputted
        if isinstance(N, np.ndarray):
            awgn = N.copy()
        else:
            awgn_real = np.random.standard_normal(
                size=(self.M, self.nbr_realizations, self.K, self.L, pilot_reuse_factor))
            awgn_img = np.random.standard_normal(
                size=(self.M, self.nbr_realizations, self.K, self.L, pilot_reuse_factor))
            awgn = np.sqrt(0.5) * (awgn_real + 1j*awgn_img)

        # Generate H_hat place holder
        H_hat = np.zeros_like(H)
        C = np.zeros_like(R)

        # Go through all BSs
        for j in range(self.L):
            # Go through all pilot groups:
            for f in range(pilot_reuse_factor):
                cur_pilot = f + 1
                # Extract the indices in pilot_ptrn that belongs to this pilot group
                group_indices = np.where(pilot_ptrn == cur_pilot)[0]

                # Compute the received pilot signal from all UEs using this pilot group to jth BS as eq (3.5)
                # The output should be a matrix of shape M x NbrOfRealizations
                yp_j = np.sqrt(ul_power) * tau_p * np.sum(H[:, :, :, cur_pilot == pilot_ptrn, j], axis=3) + np.sqrt(
                    tau_p) * awgn[:, :, :, j, f]  # awgn is scaled to its std
                if self.debug:
                    assert yp_j.shape == (self.M, self.nbr_realizations, self.K), "yp_j is not of shape {}x{}x{}".format(
                        self.M, self.nbr_realizations, self.K)
                # Go through all UEs
                for k in range(self.K):
                    # Compute the inverted version of Psi for kth user in (3.10)
                    inverted_Psi_k_j = ul_power * tau_p * \
                        np.sum(R[:, :, k, cur_pilot == pilot_ptrn, j],
                               axis=2) + eye_M
                    Psi_k_j = np.linalg.inv(inverted_Psi_k_j)
                    if self.debug:
                        assert np.allclose(
                            np.dot(Psi_k_j, inverted_Psi_k_j), eye_M), "Psi_k_j is not inverted correctly"
                    # Go through all cells that use the same pilot
                    for l in group_indices:
                        # Compute MMSE estimate of channel between BS j and UE k in cell l using (3.9) in Theorem 3.1
                        RPsi = R[:, :, k, l, j].dot(Psi_k_j)
                        H_hat[:, :, k, l, j] = np.sqrt(
                            ul_power) * RPsi.dot(yp_j[:, :, k])

                        # Compute corresponding estimation error correlation matrix, using (3.11)
                        C[:, :, k, l, j] = R[:, :, k, l, j] - \
                            (ul_power * tau_p * RPsi.dot(R[:, :, k, l, j]))
        return tau_p, H_hat, C

    def get_DL_SINR_comps(self, H, H_hat, C, tau_c, tau_p, ul_power):
        '''
        Derrive a_{jk} and b_{lijk} in DL SINR from the UL combining vectors computed from
        channel estimation tensor H_hat and its error correlation tensor C
        '''
        # Store identity matrices of different sizes
        eye_M = np.eye(self.M)

        # Compute the sum of all estimation error correlation matrices corresponds to
        # channel estimation from eacb kth UE in each lth cell to jth BS. Since there
        # are L BSs, the result should be a tensor of shape M x M x L. See equ (4.4)
        C_total_M = ul_power * np.sum(C, axis=(2, 3))

        # Compute the prelog factor assuming only downlink transmission
        prelog_factor = (tau_c-tau_p)/(tau_c)

        # Prepare to store simulation results for desired signal gains
        signal_MR = np.zeros((self.K, self.L), dtype=complex)
        signal_MMMSE = np.zeros_like(signal_MR)

        # Prepare to store simulation results for inteference signal powers
        interf_MR = np.zeros((self.K, self.L, self.K, self.L), dtype=complex)
        interf_MMMSE = np.zeros_like(interf_MR)

        # Go through all channel realizations
        for n in range(self.nbr_realizations):
            # Go through all BSs to compute matrices of combining vectors in UL direction
            for j in range(self.L):
                # Extract nth channel realization from all UEs to jth BS. The result
                # should be a tensor of shape M(arrays) x K(UEs) x L(cells) and reshaped into a matrix of shape
                # (M, K*L). The transpose order is to match Matlab's reshape rule
                H_j = np.transpose(
                    H[:, n, :, :, j], (0, 2, 1)).reshape((100, 5 * 4))

                # Same operation on H_hat
                H_hat_j = np.transpose(H_hat[:, n, :, :, j], (0, 2, 1)).reshape(
                    (self.M, self.K * self.L))

                # Compute MR combining matrix in equ (4.11)
                # Select K users in the cell containing jth BS
                V_MR_j = H_hat_j[:, self.K*j:self.K*(j+1)]
                if self.debug:
                    assert V_MR_j.shape == (
                        self.M, self.K), "MR combining matrix of {}th cell is not of shape ({},{})".format(j, self.M, self.K)
                # Compute MMMSE combining matrix in equ (4.7), note that sigma_{UL}^2 = 1
                V_MMSE_j = np.linalg.inv(ul_power * np.dot(H_hat_j, H_hat_j.conj().T)
                 + C_total_M[:, :, j] + eye_M).dot(V_MR_j / ul_power)  # Why divide?
                if self.debug:
                    assert V_MMSE_j.shape == (
                        self.M, self.K), "MMMSE combining matrix of {}th cell is not of shape ({},{})".format(j, self.M, self.K)
                # Go through all UEs in jth cell to compute corresponding precoding matrix
                # w_{jk} = \frac {v_{jk}}{\norm{v_{jk}}}
                for k in range(self.K):
                    if np.linalg.norm(V_MMSE_j[:, k]) > 0:
                        # M-MMSE precoding for kth UE in jth cell
                        w_jk = V_MMSE_j[:, k]/np.linalg.norm(V_MMSE_j[:, k])
                        # reshape into a column vector
                        w_jk = np.expand_dims(w_jk, axis=1)

                        # Compute the components of the expectations of the signal term (a_{jk} in (7.2))
                        signal_MMMSE[k, j] += np.dot(w_jk.conj().T, np.expand_dims(
                            H[:, n, k, j, j], axis=1))/self.nbr_realizations

                        # Compute the components of the expectations of the interference term
                        # (b_{lijk} in 7.3) without the substracting part if (l,i)==(j,k)
                        interf_MMMSE[k, j, :, :] += np.reshape(
                            np.abs(np.dot(w_jk.conj().T, H_j))**2, (self.L, self.K)).T/self.nbr_realizations

        # Element-wise square signal_X terms to retrieve a_{jk} in (7.2)
        signal_MMMSE = (np.abs(signal_MMMSE))**2

        # Substract the desired signal in interf_X terms to retrieve b_{lijk} in (7.3)
        for j in range(self.L):
            for k in range(self.K):
                interf_MMMSE[k, j, k, j] -= signal_MMMSE[k, j]

        # Convert to real numbers
        interf_MMMSE = np.abs(interf_MMMSE)
        return signal_MMMSE, interf_MMMSE, prelog_factor

    def compute_DL_SE(self, dl_power, desired_signal, interference, prelog_factor):
        '''
        Compute the SE in Theorem 4.6 using the formulation in (7.1)

        INPUT:
            dl_power          : K x L matrix where element (k,j) is the downlink transmit
                                power allocated to UE k in cell j
            desired_signal    : K x L matrix where element (k,j) is a_jk in (7.2)
            interference      : K x L x K x L matrix where (l,i,j,k) is b_lijk in (7.3)
            prelog_factor     : Downlink sequence length to coherence block length ratio, or tau_d/tau_c
        OUTPUT:
            SE                : K x L  matrix where element (k,j) is the downlink SE of UE k in cell j
        '''
        SE = np.zeros((self.K, self.L))  # real numbers
        for j in range(self.L):
            for k in range(self.K):
                SE[k, j] = prelog_factor * np.log2(1 + ((dl_power[k, j] * desired_signal[k, j]) / (
                    np.sum(dl_power * interference[:, :, k, j]) + 1)))

        return SE