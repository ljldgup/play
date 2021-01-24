

cpu = manager:machine().devices[":maincpu"]
mem = cpu.spaces["program"]
frames = 1
freq = 30
restarted = false
s = ""

for tag, port in pairs(manager:machine():ioport().ports) do
	if port.fields["1 Player Start"] then
	  start =  {desc =  "1 Player Start", field = port.fields["1 Player Start"]}
	end
	if port.fields["P1 Up"] then
	  up =  {desc =  "P1 Up", field = port.fields["P1 Up"]}
	end
	if port.fields["P1 Down"] then
	 down =  {desc =  "P1 Down", field = port.fields["P1 Down"]}
	end
	if port.fields["P1 Left"] then
	  left =  {desc =  "P1 Left", field = port.fields["P1 Left"]}
	end
	if port.fields["P1 Right"] then
	  right =  {desc =  "P1 Right", field = port.fields["P1 Right"]}
	end
	if port.fields["P1 Button 1"] then
	  button1 =  {desc =  "P1 Button 1", field = port.fields["P1 Button 1"]}
	end
	if port.fields["P1 Button 2"] then
	  button2 =  {desc =  "P1 Button 2", field = port.fields["P1 Button 2"]}
	end
	if port.fields["P1 Button 3"] then
	  button3 =  {desc =  "P1 Button 3", field = port.fields["P1 Button 3"]}
	end
	if port.fields["P1 Button 4"] then
	  button4 =  {desc =  "P1 Button 4", field = port.fields["P1 Button 4"]}
	end
	if port.fields["Freeze"] then
		pause =  {desc =  "Pause", field = port.fields["Freeze"]}
	end
end


action_buttons = {}
action_buttons[1] =  {left, down}
action_buttons[2] =  {down}
action_buttons[3] =  {right, down}
action_buttons[4] =  {left}
action_buttons[5] = {}
action_buttons[6] =  {right}
action_buttons[7] =  {up, left}
action_buttons[8] =  {up}
action_buttons[9] =  {up, right}
action_buttons[10] =  {button1}
action_buttons[11] = {button2}
action_buttons[12] = {button3}
action_buttons[13] = {button4}
action_buttons[14] =  {button1, button2}
action_buttons[15] = {button3, button4}
action_buttons[16] =  {button1, button2, button3}
action_buttons[17] =  {start}
action_buttons[18] = {pause}

remain_directions = {}
remain_actions = {}

--function后面第一行空，在命令行会导致incomplete command,if for也一样
function operation(act_num)
    -- 无论方向还是动作，都必须松开之前的动作键位
    print("action relase")
    for i = 1, #remain_actions do
       --print(release_buttons[i].desc)
       remain_actions[i].field:set_value(0)
    end
    remain_actions = {}

    if act_num < 10 then
        direction(act_num)
    else
        action(act_num)
    end
end

function direction(act_num)
    press_buttons = action_buttons[act_num]
    -- for i = 1, #remain_directions do
       -- print("release_buttons", remain_directions[i].desc)
       -- remain_directions[i].field:set_value(0)
    -- end
    for i = 1, #press_buttons do
        print("press_buttons", press_buttons[i].desc)
        press_buttons[i].field:set_value(1)
    end

    for i = 1, #remain_directions do
        exists=false
        for j = 1, #press_buttons do
            if press_buttons[j] == remain_directions[i] then
                print("reamin", press_buttons[j])
                exists = true
                break
            end
        end

        if not exists then
            remain_directions[i].field:set_value(0)
        end
    end

    remain_directions = press_buttons
end


function action(act_num)
    press_buttons = action_buttons[act_num]
    for i = 1, #press_buttons do
       print("press_buttons", press_buttons[i].desc)
        press_buttons[i].field:set_value(1)
    end

   remain_actions = press_buttons
end

 function all_release()
    for tag, port in pairs(manager:machine():ioport().ports) do
        for i, button in pairs(port.fields) do
            print(i,"release")
            button:set_value(0)
        end
   end
end

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
            -- 设定成固定人物
            p1 = mem:read_i8(0x10A84e)
            p2 = mem:read_i8(0x10A861)

            if p2 > 0 or p1 ~= 27
            then
                mem:write_i8(0x10A85f, 0)
                mem:write_i8(0x10A860, 0)
                mem:write_i8(0x10A861, 0)
                --改颜色
                mem:write_i8(0x10A862, 1)

                -- 貌似这里设置有时会导致选人卡顿，所以脚本直接一起选
                mem:write_i8(0x10A84E, 27)
                mem:write_i8(0x10A84F, 27)
                mem:write_i8(0x10A850, 27)

                --改颜色
                mem:write_i8(0x10A851, 0)
            end

            countdown = mem:read_i16(0x10A83A)
            if countdown ~= 0 and countdown ~= 24626
            then
                -- act code
                act1 = mem:read_i16(0x108172)
                act2 = mem:read_i16(0x108372)

                --12p xy坐标
                x1 = mem:read_i16(0X108118)
                y1 = mem:read_i8(0X108121)
                x2 = mem:read_i16(0X108318)
                y2 = mem:read_i8(0X108321)

                --energy
                energy1 = s..mem:read_i8(0x1082E3)
                energy2 = s..mem:read_i8(0x1084E3)
                --1p曝气
                baoqi = mem:read_i8(0x1081E0)//16
                --2p爆气
                baoqi2 = mem:read_i8(0x1083E0)//16

                --1p人物
                role1 = mem:read_i8(0x108171)
                --2p人物
                role2 = mem:read_i8(0x108371)
                --1p的破防值作为计算防御报酬使用
                guard_value1 = mem:read_i8(0x108247)

                --连击，数血量作为reward
                count1 = mem:read_i8(0x1084CE)

                life1 = mem:read_i8(0x108239)
                life2 = mem:read_i8(0x108439)

                --时间用来结合血量判断状态，用于生成reward
                --币数用来判断是否输掉，家用机game币数会回到4
                print(act1, act2, x1, y1, x2, y2, energy1, energy2, baoqi, baoqi2, role1, role2, guard_value1, count1, life1, life2,countdown, coin)

                action_num = io.read("*num")
				operation(action_num)
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
