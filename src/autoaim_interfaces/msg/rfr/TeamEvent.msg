bool restoration_1  # bit 0: 己方与兑换区不重叠的补给区占领状态，1 为已占领
bool restoration_2  # bit 1: 己方与兑换区重叠的补给区占领状态，1 为已占领
# bit 2: 己方补给站的占领状态，1 为已占领 @attention 仅 RMUL 适用 */
bool supplier_area
bool small_power_rune  # bit 3: 己方小能量机关的激活状态，1 为已激活
bool large_power_rune  # bit 4: 己方大能量机关的激活状态，1 为已激活
# bit 5-6: 己方中央高地的占领状态，1 为被己方占领，2 为被对方占领 */
uint8  central_highland
# 7-8: 己方梯形高地的占领状态，1 为已经占领*/
uint8  trapezoid_highland

# bit 9-17: 对方飞镖最后一次击中己方前哨站或基地的时间,0-420，开局默认为 0
uint16  dart_last_hit_us_time
# bit 18-20: 对方飞镖最后一次击中己方前哨站或基地的具体目标,开局默认为 0, 1 为击中前哨站，2 为击中基地固定目标，3 为击中基地随机固定目标，4 为击中基地随机移动目标
uint8  dart_last_hit_us_type 
# bit 21-22: 中心增益点的占领情况, 0 为未被占领，1 为被己方占领，2 为被对方占领，3 为被双方占领。 @attention 仅 RMUL 适用
uint8  center_buff_area