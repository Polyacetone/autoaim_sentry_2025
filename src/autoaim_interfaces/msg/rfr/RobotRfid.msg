bool our_base                  # bit 0: 己方基地增益点
bool our_central_highland      # bit 1: 己方中央高地增益点
bool opp_central_highland      # bit 2: 对方中央高地增益点
bool our_trapezoid_highland    # bit 3: 己方梯形高地增益点
bool opp_trapezoid_highland    # bit 4: 对方梯形高地增益点
bool our_launch_front          # bit 5: 己方飞坡增益点（靠近己方一侧飞坡前）
bool our_launch_back           # bit 6: 己方飞坡增益点（靠近己方一侧飞坡后）
bool opp_launch_front          # bit 7: 对方飞坡增益点（靠近对方一侧飞坡前）
bool opp_launch_back           # bit 8: 对方飞坡增益点（靠近对方一侧飞坡后）
bool our_highland_bottom       # bit 9: 己方地形跨越增益点（中央高地下方）
bool our_highland_top          # bit 10: 己方地形跨越增益点（中央高地上方）
bool opp_highland_bottom       # bit 11: 对方地形跨越增益点（中央高地下方）
bool opp_highland_top          # bit 12: 对方地形跨越增益点（中央高地上方）
bool our_highway_bottom        # bit 13: 己方地形跨越增益点（公路下方）
bool our_highway_top           # bit 14: 己方地形跨越增益点（公路上方）
bool opp_highway_bottom        # bit 15: 对方地形跨越增益点（公路下方）
bool opp_highway_top           # bit 16: 对方地形跨越增益点（公路上方）
bool our_fort                  # bit 17: 己方堡垒增益点
bool our_outpost               # bit 18: 己方前哨站增益点
bool our_restoration_1         # bit 19: 己方与兑换区不重叠的补给区/RMUL 补给区 
bool our_restoration_2         # bit 20: 己方与兑换区重叠的补给区
bool our_big_resource_island   # bit 21: 己方大资源岛增益点
bool opp_big_resource_island   # bit 22: 对方大资源岛增益点
bool central_boost             # bit 23: 中心增益点 @attention 仅RMUL适用