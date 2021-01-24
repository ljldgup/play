cpu = manager:machine().devices[":maincpu"]
mem = cpu.spaces["program"]


----------------------------------------------------------------------------------------------------------------------
--这里是获取按键对象

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

remain_direction= 5
remain_action = 5

--function后面第一行空，在命令行会导致incomplete command,if for也一样
--如果程序没按预期走，也没报错，就看看是不是所有函数都被正确定义了
function operation(num)
    -- 无论方向还是动作，都必须松开之前的动作键位
    --print("action relase")
    remain_actions_button = action_buttons[remain_action]
    for i = 1, #remain_actions_button do
       --print(release_buttons[i].desc)
       remain_actions_button[i].field:set_value(0)
    end
    if num < 10 then
        direction(num)
        --如果是方向，直接重置动作
        remain_action = 5
    else
        action(num)
    end
end

function direction(direction_num)
    if direction_num ~= remain_direction then
        press_buttons = action_buttons[direction_num]
        remain_butttons = action_buttons[remain_direction]
        for i = 1, #press_buttons do
            --print("press_buttons", press_buttons[i].desc)
            press_buttons[i].field:set_value(1)
        end
        for i = 1, #remain_butttons do
            exists=false
            for j = 1, #press_buttons do
                if press_buttons[j] == remain_butttons[i] then
                    --print("reamin", press_buttons[j])
                    exists = true
                    break
                end
            end
            if not exists then
                remain_butttons[i].field:set_value(0)
            end
        end
        remain_direction = direction_num
    end
end


function action(act_num)
    --一样的键不停按会导致失效，这里暂停一次
    if act_num == remain_action then
        remain_action = 5
    else
        press_buttons = action_buttons[act_num]
        remain_buttons = action_buttons[remain_action]
        for i = 1, #press_buttons do
           --print("press_buttons", press_buttons[i].desc)
            press_buttons[i].field:set_value(1)
        end
       remain_action = act_num
    end
end

 function all_release()
    for tag, port in pairs(manager:machine():ioport().ports) do
        for i, button in pairs(port.fields) do
            --print(i,"release")
            button:set_value(0)
        end
   end
end



----------------------------------------------------------------------------------------------------------------------
--这里开始是运行状态,

restarted = false
restart_step = 1000
restart_buttons={start, start, down, down}

for i=5, 200, 1 do
    restart_buttons[i] = button1
end

total_steps = 100
step_interval = 10

function restart()
    --restarted 将restarted将每次game over后的输出限制到一次，避免输出过多造成卡死
    print("4")
    restart_step = step_interval
end

--重启选人
function restarting()
    print(restart_step)
    if restart_step % step_interval == 0 then
        print(restart_step//step_interval, restart_buttons[restart_step//step_interval].desc, "press")
        restart_buttons[restart_step//step_interval].field:set_value(1)
    end
    if restart_step % step_interval == 5 then
        print(restart_step//step_interval, restart_buttons[restart_step//step_interval].desc, "release")
        restart_buttons[restart_step//step_interval].field:set_value(0)
    end
    if restart_step > total_steps then
        --确保都松了
        button1.field:set_value(0)
        start.field:set_value(0)
        down.field:set_value(0)
    end
    restart_step = restart_step + 1
end

function running()
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
    if countdown ~= 0 and countdown ~= 24626 then
        -- act code
        act1 = mem:read_i16(0x108172)
        act2 = mem:read_i16(0x108372)

        --12p xy坐标
        x1 = mem:read_i16(0X108118)
        y1 = mem:read_i8(0X108121)
        x2 = mem:read_i16(0X108318)
        y2 = mem:read_i8(0X108321)

        --energy
        energy1 = mem:read_i8(0x1082E3)
        energy2 = mem:read_i8(0x1084E3)
        --1p曝气
        baoqi1 = mem:read_i8(0x1081E0)//16
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
        print(act1, act2, x1, y1, x2, y2, energy1, energy2, baoqi1, baoqi2, role1, role2, guard_value1, count1, life1, life2, countdown, coin)

        action_num = io.read("*num")
        operation(action_num)
    end
end


-- frames统计帧数， 每freq+1帧做一次输出
frames = 1
freq = 5

--使用 币数coin 和restart_step来确定当前状态
function func()
    frames=frames+1
    --每freq帧输出一次，freq可调，太大太小都不太合适
    if frames > freq then
        --调试输出用
        --os.execute("cls")
        coin = mem:read_i8(0x10A816)
        --print("coin", coin)
        --用掉了一个币，此时在进行游戏
        if  coin~=4 then
            if restart_step < total_steps + 2 then
                restarting()
            else
                running()
            end
        else
            if restart_step > total_steps then
                restart()
            else
                restarting()
            end
        end
        frames = 0
    end
end

emu.register_frame_done(func)
