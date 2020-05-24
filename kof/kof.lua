

cpu = manager:machine().devices[":maincpu"]
mem = cpu.spaces["program"]
frames = 1
freq = 6 
restarted = false
s = ""

function test()
    frames=frames+1
    
    --每freq帧输出一次，freq可调，太大太小都不太合适
    if(frames>freq)
    then
        --调试输出用
        --os.execute("cls")
        coin = mem:read_i8(0x10A816)
        
        --用掉了一个币，此时在进行游戏
        if  coin~=4 
        then
            -- 对手设定成固定人物
            -- p1 = mem:read_i8(0x10A84e)
            p2 = mem:read_i8(0x10A861)
            -- 貌似这里设置有时会导致选人卡顿，所以等他设完了在该，选人值为FF的情况下不改
            if p2 > 0
            then
                mem:write_i8(0x10A85f, 0)
                mem:write_i8(0x10A860, 0)
                mem:write_i8(0x10A861, 0)
                --改颜色
                mem:write_i8(0x10A862, 1)
            end
            
            countdown = mem:read_i16(0x10A83A)
            if countdown ~= 0 and countdown ~= 24626
            then
                -- act code, energy
                s = mem:read_i16(0x108172).." "
                s = s..mem:read_i16(0x108372).." "
                
                s = s..mem:read_i8(0x1082E3).." "
                s = s..mem:read_i8(0x1084E3).." "

                --12p xy坐标
                t = (mem:read_i16(0X108118)-380)/380
                s = s..t.." "
                t = (mem:read_i16(0x108120)-128)/128
                s = s..t.." "
                t = (mem:read_i16(0X108318)-380)/380
                s = s..t.." "
                t = (mem:read_i16(0x108320)-128)/128
                s = s..t.." "

                --爆气
                t = mem:read_i8(0x1083E0)//16
                s = s..t.." "
                
                --1p人物               
                s = s..mem:read_i8(0x108171).." "
                --2p人物               
                s = s..mem:read_i8(0x108371).." "
                --1p的破防值作为计算防御报酬使用
                s = s..mem:read_i8(0x108247).." "
                
                --连击，数血量作为reward
                s = s..mem:read_i8(0x1084CE).." "

                s = s..mem:read_i8(0x108239).." "
                s = s..mem:read_i8(0x108439).." "

                --时间用来结合血量判断状态，用于生成reward
                s = s..countdown.." "
                --币数用来判断是否输掉，家用机game币数会回到4
                s = s..coin
                print(s)
            end
            --币数变3后可重启
            restarted = false
        else
            if not restarted
            then
                --restarted 将restarted将每次game over后的输出限制到一次，避免输出过多造成卡死
                s = "4"
                print(s)
                restarted = true
            end
        end
        
        frames = 0
            
    end
end

emu.register_frame_done(test)
