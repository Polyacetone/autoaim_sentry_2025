uint16  allowance        # 除远程兑换外，哨兵成功兑换的发弹量
uint8  remote_allowance   # 哨兵成功远程兑换发弹量的次数
uint8  remote_hp          # 哨兵成功远程兑换血量的次数
bool allow_free_resurrection         # 哨兵当前是否可以确认免费复活
bool allow_redemption_resurrection   # 哨兵当前是否可以兑换立即复活
uint16  redemption_resurrection_cost   # 哨兵当前兑换立即复活需要花费的金币数
bool out_of_war_status         # 哨兵当前是否处于脱战状态
uint16  remain_total_allowance  # 队伍17mm允许发弹量的剩余可兑换数
