# TSQ — z=3 混合面型 4.8.8:连续广延自由、整数字母表零熵("离散化致刚")

日期:2026-09-27。状态:**定理证明完成,数值验证通过,可写入论文 SI。**
用户提议的"4.8.8 度量随机密铺性测试"**已闭合:答案是 s(∞) = 0**。

**结论先行**:4.8.8(截角方格,z=3,等角四边形(矩形)+等角八边形 1:1)是本项目第一个
**"ℝ 与 ℤ 答案相反"**的样本:

> 实(ℝ)解空间维数 = 2L²+2 = F+2(**每面 +1,广延**,与蜂巢同密度,
> 浮点秩 L=2..5 逐位确认);
> 但整数字母表下 **Ω = k^(类数) 精确,类数 = 4L(L×L 环面,L=2..12
> 全部命中,亚广延)⟹ s(∞) = 0**。

机制 = **无理数劈行**:45° 方向族的闭合投影行内混合 1 与 √2/2,整数
(或有理)取值下单条实等式 ⟺ 两条整系数等式(有理部、√2 部各自为零)。
劈行把八边形的 2 条实约束变成 **4 组对边相等**(轴对 2 组来自有理部,
对角对 2 组来自两条 √2 行相加减),全系统化为纯等式 ⟹ 传播 ⟹ 刚性。
**离散字母表探测到了方向代数的无理性**——连续自由被量化杀死。

---

## 1. 设定

- 图:4.8.8 = 方格截角构造。原方格面 → 八边形,原内部顶点 → 四边形;
  4.8.8 顶点 = 原键上两个截断点(互不相同)。边两类:轴段(八边形-
  八边形,方向 0°/90°)与截断边(四边形-八边形,方向 45/135/225/315°)。
  3 价;环面 L×L:V = 4F,E = 6F,F_sq = F_oc = F。
- 等角:四边形内角 90°、八边形 135°(顶点构型强制);**四边形 = 等角四边形(矩形)
  口径**:闭合约束只锁对边相等、从不锁邻边相等(2026-09-28 措辞勘误,原文"正方形"
  系笔误,已有全部数值本就是矩形口径,无需重跑);字母表取整数(项目口径 {4,5,6})。
- 方向代数:轴族系数 ∈ {±1, 0},对角族系数 = ±√2/2 ⟹ 同一条投影行内
  **混合**有理与 √2 系数(与 60°/90° 系统的本质差别,见 §4)。

## 2. 定理与证明

**引理 1(无理数劈行)**:设行系数 c_i = a_i + b_i√2(a_i, b_i ∈ ℚ)。
对有理取值 s_i,Σc_i s_i = 0 ⟺ Σa_i s_i = 0 且 Σb_i s_i = 0。
(有理数与无理数在 ℚ 上线性无关。)□

**引理 2(八边形劈行 = 四组对边相等)**:八边形八边方向 0..7(×45°)
环向,闭合 x/y 投影行(×2 后)为
  x: (2,0,0,0,−2,0,0,0) 与 (0,1,0,−1,0,−1,0,1)
  y: (0,0,2,0,−2,0,0,0) 与 (0,1,0,−1,0,1,0,−1)(按边序)
即有理部给 **s₁ = s₅, s₃ = s₇**(对边轴段),两条 √2 行相加减给
**s₂ = s₆, s₄ = s₈**(对边截断段)。四边形同理两轴对相等(等角四边形=矩形口径,见 §1 勘误)。
(脚本内逐面做"等式行张成 == 拆分行张成"的秩检验后再用。)□

**定理(条带参数化,Ω 精确)**:劈行后全部约束均为对边相等型等式;
对边配对传递成条带(strand,4 个方向族各若干条),解 = 条带上取常数,

> **Ω = k^(类数)** 精确;L×L 环面类数 = **4L**(数值 2..12 逐位),
> 一般 O(Lq+Lr) ⟹ s = 类数·ln k/F → 0。

**对照(ℝ 口径)**:不劈行时每面恰 2 条实投影等式,net = E−2F =
F(每面 +1,广延);浮点秩给 dim_ℝ = 2L²+2(L=2..5 逐位 = 2L²+2)。
两个口径**都是对的**——它们回答不同的问题:ℝ 维数量连续形变空间,
ℤ 计数数离散热构型;4.8.8 的连续自由住在无理方向比上,字母表取不到。

## 3. 分类学后果

1. **混合面型 z=3 族的正熵候选被关闭(对 ℤ 字母表)**:4.8.8 s=0。
   机制预言同族阵亡:**3.12.12 已验证(2026-09-27,ZCode 独立截角蜂窝
   构造:dim_ℝ=F+2、dim_ℚ=11/19/29 随 L 线性 ⟹ 亚广延零熵;回归见
   z4_3464/RTH_数值验证.py)**;4.6.12 与其余含无理面型顶点型仍未验证,
   写作时如实标注"未验证"。
2. **四图分类的算术读法**:蜂巢/方格/kagome/三角格全部是 60°/90° 族
   ⟹ 每条投影行要么纯有理、要么纯 √3 因子 ⟹ **无劈行** ⟹ 蜂巢的
   广延自由在 ℤ 上存活。45°/30° 族才有劈行。**"蜂巢 = 唯一正熵载体"
   的"ℤ 下无条件"升级已被 3.4.6.4 反例推翻(2026-09-27 DSH 审核发现、
   ZCode 独立复算证实,z4_3464/)**:劈行机制只覆盖含 p∉{3,4,6} 面的密铺
   (Niven 判据);全有理面型混合密铺 dim_ℚ=dim_ℝ,3.4.6.4 的
   dim = F_hex+3 广延(正熵强证据)。可辩护表述 = "单面型 {3,4,6} 网络
   内唯一;全有理混合族按 ℚ-零化数分类,开放"。
3. 论文 1 措辞:把"ℝ 广延 vs ℤ 零熵"作为 4.8.8 一段写入(它同时是
   "度量 random tiling 存在性由字母表算术裁决"的最干净示例)。

## 4. 数值验证(`TSQ_数值验证.py` → `TSQ_熵密度验证.csv`)

四层,**ALL CHECKS PASSED**:
1. DFS 暴力锚点(2×2 环面 k=2,逐面劈行约束剪枝)= 核枚举(256=256);
2. 条带 TM ↔ 核枚举 ↔ 3^类数 三方逐位互检(拼块 (2,2)/(3,2)/(2,3)
   k=3:59049/1594323/1594323,全为 3 的纯幂);
3. 环面精确 Ω 系列全为 3 的纯幂((3,3): 3¹²;类数 4L:8,12,16,20,
   24,32,40,48 对 L=2..12);s = 4L·ln3/L² → 0;
4. ℝ-ℤ 对照:dim_ℝ = 2L²+2(L=2..12 零违规)vs 类数 = 4L;
   拼块系列 (32×32): 类数 190,F=1985,s=0.105;(60×30): 0.084。

## 5. 英文 SI 成文草稿(SI-ready)

> **Theorem (discretization-induced rigidity of the 4.8.8 tiling).**
> Let G be the L×L truncated-square (4.8.8) graph on a torus, faces
> equiangular quadrilaterals (rectangles) and equiangular octagons in
> ratio 1:1, all angles fixed by the
> isogonal condition, edge lengths drawn from a finite alphabet A ⊂ ℤ.
> Then Ω = k^{4L} exactly and s = 4L ln k /(2L²) → 0 — even though the
> real solution space has extensive dimension dim_ℝ = 2L²+2.
>
> **Proof.** The closure projections of the octagon mix coefficients 1
> and √2/2. For integer (rational) edge values, each such equation
> Σ(a_i+b_i√2)s_i = 0 is equivalent to the pair Σa_i s_i = 0,
> Σb_i s_i = 0 (rational independence). The split octagon rows combine
> into four opposite-side equalities; squares contribute two. Every
> constraint is thus an equality; opposite-pairing propagates along the
> four direction families into strands, solutions are constants on
> strands, and Ω = k^{#strands} = k^{4L}. Meanwhile the un-split real
> system retains one free variable per face (dim_ℝ = 2L²+2): the
> continuous freedom lives on irrational direction ratios, which no
> integer alphabet can reach. □
>
> Consequence: the mixed-face z=3 family (4.8.8, 3.12.12, 4.6.12) is
> excluded as a discrete-alphabet metric-random-tiling carrier; the
> honeycomb's uniqueness holds unconditionally over integer alphabets.
> The four canonical graphs (60°/90° direction algebra) are split-free —
> each closure projection is purely rational or a pure √3 multiple —
> which is precisely why their entropy classification survives
> discretization.

## 6. 建模教训

1. **√2 行拆分双矩阵**:计数用拆分整行(整数点精确等价),维数用
   浮点 cos/sin 行——拆分行在 ℝ 上秩虚高,绝不可混用(库 docstring
   已写死此警告)。
2. **内核枚举预算闸**:3^20 级 `list(product(...))` 会以数百 GB 需求
   拖垮系统(本次 ZCode 崩溃的直接原因);tsq/kernel 枚举一律先验
   k^dim × 列宽 ≤ 2e8,超限拒绝。
3. 小机器纪律(PROGRESS 2026-09-23 已有)再次生效:GB 级数组一律
   先算账。

## 7. 文件与状态

- 本文档 + `tsq_lib.py` + `TSQ_数值验证.py` + `TSQ_熵密度验证.csv`
  (本目录)。数据地图前缀 TSQ_,目录 z3_488/。
- 待办:并入论文 1 分类节(ℝ-ℤ 对照段 SI 可直接取用 §5);
  3.12.12/4.6.12 的逐一验证为可选项(机制已同)。
