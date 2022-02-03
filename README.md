工作流程

一 扫码确定当前计量柜cibucle_id
二 从数据库读取当前计量柜包含所有端子及安装单位信息
三 拍摄图片
	1 对于图片中的所有rec（包括terminal端子和plate安装单位铭牌）
	TODO：PCA，数据输入顺序改进
	2 按y升序重排（对于两列端子，倾斜特别严重时可能会错判）
	TODO：只对每个plate所属端子进行PCA分组，不属于同一个plate下的肯定不在一组。同一个plate下的可能不在一组
	3 通过PCA，将端子分组
	4 对每组端子，进行基于回归的terminal形状位置数据矫正；plate样本少，保持原数据
	5 依据矫正后的形状位置数据，裁切terminal的中心文本部分
	TODO：Rec实例也应该具有group属性，从而方便对识别返回数据快速分组
	6 拼接所有裁切部分，调用api识别；plate不进行拼接（使用Rec实例，具有属性xy_list和classes）
	TODO：对于plate识别数据，可以采取矫正手段，例如在当前计量柜安装单位信息中选取最接近的作为结果
	7 识别结果为端子编号terminal_num和安装单位铭牌文本plate_text
四 查询数据库
	1 根据plate_text查询安装单位id：install_id
	2 识别结果端子编号terminal_num，组合cibucle_id和install_id生成变电站中每个端子独一无二的terminal_id
	3 查询terminal_id所连回路loop_num
	4 在原始图片中标注符合要求的terminal