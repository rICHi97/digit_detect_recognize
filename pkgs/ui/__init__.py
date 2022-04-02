# -*- coding: utf-8 -*-
"""
Created on 2022-02-28 01:28:28

@author: Li Zhi
在ui.qt_designer_code中通过eric定义ui布局，命名为ui_window.py
在ui.window中通过导入qt_code，定义信号与槽，命名为window.py
在ui.ui_app中集成多个window作为app，命名为app.py
app.py中可以定义多个app类
在pkgs.app中实例一个app类
TODO：ui的相关工具代码需要修改，比如从eric更新qt_code，以及生成资源文件等
TODO：ui代码之后完善，先完成简单工具雏形
"""
