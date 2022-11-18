    # def collision_check(self, x, y, th):
    #     """ Get edge vectors of the robot body,
    #         and check collisions on edges
    #         anti-clock wise
    #         p1->p2 (lendth), p2->p3 (width)
    #         p3->p4 (lendth), p4->p1 (width)
    #     """
    #     p1, p2, p3, p4 = self.calc_rect_vertices(x, y, th)
    #     print("Checking collsion at: ", [x,y,th])
    #     for point in [p1,p2,p3,p4]:
    #         ir = point[1] # row index
    #         ic = point[0] # col index
    #         if self. check_point_occupancy(ir, ic) == True:
    #             print ("Collision Detected !")
    #             return True

    #     l1 = self.build_line(p1, p2)
    #     for point in l1:
    #         ir = point[1] # row index
    #         ic = point[0] # col index
    #         if self. check_point_occupancy(ir, ic) == True:
    #             print ("Collision Detected !")
    #             return True
    #     l2 = self.build_line(p2, p3)
    #     for point in l2:
    #         ir = point[1] # row index
    #         ic = point[0] # col index
    #         if self. check_point_occupancy(ir, ic) == True:
    #             print ("Collision Detected !")
    #             return True
    #     l3 = self.build_line(p3, p4)
    #     for point in l3:
    #         ir = point[1] # row index
    #         ic = point[0] # col index
    #         if self. check_point_occupancy(ir, ic) == True:
    #             print ("Collision Detected !")
    #             return True
    #     l4 = self.build_line(p4, p1)
    #     for point in l4:
    #         ir = point[1] # row index
    #         ic = point[0] # col index
    #         if self. check_point_occupancy(ir, ic) == True:
    #             print ("Collision Detected !")
    #             return True
    #     print (" No collision detected :)")
    #     return False


        # def recovery_state_update(self, state, d_t):
    #     """ Recovery state update when collsion happens"""
    #     s = state ; dt =d_t; collision = True
    #     ss = self.calc_state(s, dt/2)
    #     if ss[5] == False:
    #         collision = False # Success
    #         return


    # def state_update(self):
    #     """
    #     Update the robot state within the acceleration limits
    #     """
    #     if self. robot_drive_type == "differential":
    #         recover = False
    #         # Old state, copy the old state for later recovery use
    #         s0,s1,s2,s3,s4,s5 = self.state[0],self.state[1],self.state[2],self.state[3],self.state[4],self.state[5]
    #         print("check s0 to s5")
    #         print(s0,s1,s3,s4,s5)
    #         print(self.state)
    #         # Calculate excuted signal within acceleration limits
    #         # Linear acceleration check
    #         if self.control_signal[0] < s3-self.a_limits[0]:
    #             self.excecut_signal[0] = s3 -self.a_limits[0]
    #         elif self.control_signal[0]  > s3 + self.a_limits[0]:
    #             self.excecut_signal[0] = s3 + self.a_limits[0]
    #         else:
    #             self.excecut_signal[0] = self.control_signal[0]
    #         # Angular acceleration check
    #         if self.control_signal[1] - s4 < -self.a_limits[1]:
    #             self.excecut_signal[1] = s4 -self.a_limits[1]
    #         elif self.control_signal[1] - s4 > self.a_limits[1]:
    #             self.excecut_signal[1] = s4 + self.a_limits[1]
    #         else:
    #             self.excecut_signal[1] = self.control_signal[1]

    #         print(" self state: ",self.state)
    #         # Update state
 
    #         ss0 = s0 + math.cos(s2)* s3 * self.dt  # update x with old speed s[3]
    #         ss1 = s1 + math.sin(s2)* s3  * self.dt  # update y with old speed
    #         ss2 = s2 + s4 * self.dt                           # update yaw angle theta
    #         ss2 = math.atan2(math.sin(s2), math.cos(s2))# bound [-pi, pi]
    #         ss3 = self.excecut_signal[0] # linear speed
    #         ss4 = self.excecut_signal[1] # rotation speed
    #         ss5 = self.body_collision_check(ss0, ss1, ss2)
    #         if ss5 == True:
    #             # self. recovery_state_update(self. old_state)
    #             # print(" Recovery finished")
    #             recovery = True
    #         else:
    #             recovery = False
    #         if recovery == True:
    #             print("recovery started ..............................................")
    #             self.state[3:] = [0,0,1]
    #         else:
    #             print("ss assigned")
    #             self.state = [ss0,ss1,ss2,ss3,ss4,ss5]
    #         self.print_robot_info()

    # def recovery_state_update(self, old):
    #     print("Recovery state update started ...................")
    #     new_state = old # old state is the state at dt*0
    #     old_state = old
    #     print("old state used in recovery: ", old_state)
    #     # Check dt/2
    #     print("Checking 1/2 dt")
    #     new_state[0] = old_state[0] + math.cos(old_state[2])* old_state[3] * self.dt/2  # update x with old speed s[3]
    #     new_state[1] = old_state[1] + math.sin(old_state[2])* old_state[3] * self.dt/2  # update y with old speed
    #     new_state[2] = old_state[2] + old_state[4] * self.dt/2                           # update yaw angle theta
    #     new_state[2] = math.atan2(math.sin(new_state[2]), math.cos(new_state[2]))# bound [-pi, pi]
    #     half_dt_state = new_state
    #     # Check whether collide at dt/2
    #     if self.body_collision_check(new_state[0], new_state[1], new_state[2]) == True:
    #         # Collide at dt/2
    #         # Check dt/4
    #         print("Checking 1/4 dt")
    #         new_state[0] = old_state[0] + math.cos(old_state[2])* old_state[3] * self.dt/4  # update x with old speed s[3]
    #         new_state[1] = old_state[1] + math.sin(old_state[2])* old_state[3] * self.dt/4  # update y with old speed
    #         new_state[2] = old_state[2] + old_state[4] * self.dt/4                           # update yaw angle theta
    #         new_state[2] = math.atan2(math.sin(new_state[2]), math.cos(new_state[2]))# bound [-pi, pi]
    #         # Check whether collide at dt/4
    #         # ***************** Case 1: dt*0 ***************************
    #         if self.body_collision_check(new_state[0], new_state[1], new_state[2]) == True:
    #             print("Collide at dt/4, use old state to update self.state") 
    #             self.state[:3] = old_state[:3]
    #             self.state[3:] = [0.0,0.0,1] # update speed as 0 and bool collision as 1
    #         # ***************** Case 2: dt*(1/4) ***************************
    #         else:
    #             print("Not collide at dt/4, use new_state at dt/4 to update self.state")
    #             self.state[:3] = new_state[:3]
    #             self.state[3:] = [0.0,0.0,1]
    #     else:
    #         # Not collide at dt/2
    #         # Check dt*3/4
    #         print("Checking 3/4 dt")
    #         new_state[0] = old_state[0] + math.cos(old_state[2])* old_state[3] * self.dt*3/4  # update x with old speed s[3]
    #         new_state[1] = old_state[1] + math.sin(old_state[2])* old_state[3] * self.dt*3/4  # update y with old speed
    #         new_state[2] = old_state[2] + old_state[4] * self.dt*3/4                           # update yaw angle theta
    #         new_state[2] = math.atan2(math.sin(new_state[2]), math.cos(new_state[2]))# bound [-pi, pi]
    #         # ***************** Case 3: dt*(1/2) ***************************
    #         if self.body_collision_check(new_state[0], new_state[1], new_state[2]) == True:
    #             print("Collide at 3/4 dt , use old state to update self.state")
    #             self.state[:3] = half_dt_state[:3]
    #             self.state[3:] = [0.0,0.0,1] # update speed as 0 and bool collision as 1
    #         # ***************** Case 4: dt*(3/4) ***************************
    #         else:
    #             print("Not collide at 3/4 dt, use new_state at 3/4 dt to update self.state")
    #             self.state[:3] = new_state[:3]
    #             self.state[3:] = [0.0,0.0,1]