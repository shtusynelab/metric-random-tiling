# KAG — z=4 kagome:等角度量无序的零熵支(解析定理 + 数值验证)

日期:2026-09-27。状态:**定理证明完成,数值验证通过,可写入论文 SI。**
四图分类的最后未闭合支,**本轮闭合**:kagome 站在 s = 0 一侧。

**结论先行**:kagome 等角网格(4 价,三角形+六边形面 = 2:1)上不存在度量
random tiling。等角约束把每个三角形三边绑定成单变量 t(蜂巢顶点上),
每个六边形给 2 条 pair-sum 型闭合等式;约束矩阵恰为方阵(2F×2F,临界)。
Bloch 分解给出符号行列式的**精确因式分解**

> **det M̂(k) = 3·(a−1)(b−1)(a−b)/a²b,a = e^{ik₁}, b = e^{ik₂}**

零点集 = k 环面上三根回绕线 {k₁=0} ∪ {k₂=0} ∪ {k₁=k₂} ⟹ 解空间维数

> **dim ker M(Lq,Lr) = Lq + Lr + gcd(Lq,Lr) − 1(精确,任意尺寸)**

亚广延(O(Lq+Lr))⟹ Ω ≤ k^dim ⟹ 每面熵密度 **s(∞) = 0**。这是**零熵的
严格定理**,不是数值外推。核可以完全显式写出:三族"条带模"(行/列/
对角),见定理 3——kagome 与方格一样"零熵 ≠ 零模",但模更少、且带有
二部交错扭转。

---

## 1. 设定与框架约定

与蜂巢/方格/三角格框架(AGENTS.md 概念约定)完全同构,仅换图:

- **组合拓扑锁定**:连通 patch 取标准 kagome 图,即蜂巢的 medial graph
  (截角构造):kagome 顶点 = 蜂巢键(每键中点),kagome 边 = 蜂巢顶点处
  相邻两键的夹角块。每条 kagome 边恰属于一个三角形(蜂巢顶点处)与一个
  六边形(蜂巢面)。记账(环面 Lq×Lr 个六边形,F = Lq·Lr):V = 3F(= 蜂巢
  键数),E = 6F,面 = 三角形 2F + 六边形 F,4 价,Euler V−E+F = 0。
  要求 Lq, Lr ≥ 2(=1 时六边形与自身共边,平行键退化——与三角格同款教训)。
- **角刚边柔**:三角形内角全钉 60°,六边形内角全钉 120°;顶点周角
  60°×2 + 120°×2 = 360°。每条边一个长度,取自离散字母表 A(|A| = k,
  项目口径 k = 3,边色/边长 4/5/6)。
- **几何观察(定向结构)**:每个 kagome 顶点的 4 条边合成两条共线对
  (每方向族一条直线穿过),边方向锁进 3 个固定族——蜂巢 ψ6≡1 取向
  相干的直接推广。

## 2. 定理与证明

**引理 0(等边绑定,精确约化)**:三角形三边分属 3 个固定方向族、内角
60°,闭合条件 Σ Lᵢwᵢ = 0(wᵢ 为三族单位向量,两两 120°)有唯一解
L₁ = L₂ = L₃。故每个三角形贡献**一个**变量 t(= 其所在蜂巢顶点上的
边长),且该约化与全边变量系统精确等价(解集双射)。□

**六边形约束**:六边形第 k 条边 = 第 k 个角上的夹角块,长度 = 该角
蜂巢顶点的 t;边方向沿周界逐边转 60°。闭合条件 = 2 条独立线性等式,
pair-sum 型(与蜂巢同型,如 t₁+t₂ = t₄+t⁵ 与 t₂+t₃ = t₅+t₆;
代码内以环向方向的整系数投影实现,两基等价)。于是约束系统约化为
**2F×2F 方阵 M**(行 = 每六边形 2 条,列 = 蜂巢顶点 t)——净自由度
0/面(临界)的直接体现。

**定理 1(符号行列式因式分解)**:取一个六边形为基胞(2 个变量:A/B
子格),六角归属 cell 偏移 δ = (0,0),(0,0),(−1,+1),(−1,0),(−1,0),
(0,−1),环向边方向的整系数投影给出行向量,得 2×2 符号 M̂(k):

    M̂[x,A] = −1 − a⁻¹b + 2a⁻¹     M̂[x,B] = −2 + a⁻¹ + b⁻¹
    M̂[y,A] =  2 − a⁻¹b − a⁻¹      M̂[y,B] =  1 − 2a⁻¹ + b⁻¹

直接展开(或数值验证,残差 < 1e−14):

    det M̂ = 3(a−1)(b−1)(a−b) / a²b。

a=1, b=1, a=b 三根即零点集 = 三根回绕线 {k₁=0} ∪ {k₂=0} ∪ {k₁=k₂}。
□(展开式与 `KAG_数值验证.py` B 层断言双重锁定。)

**定理 2(零化数精确公式)**:M 与环面平移对易 ⟹ 按 Bloch 动量分解,
dim ker M = Σ_{k ∈ 网格} nullity M̂(k)。网格点落在零点集上者:
{k₁=0} 有 Lr 个、{k₂=0} 有 Lq 个、{k₁=k₂} 有 gcd(Lq,Lr) 个;并集去掉
原点重复 = Lq+Lr+g−2 个。原点处 M̂(0) = 0 零矩阵(零化数 2:常向量 +
二部交错向量),其余零点 2×2 符号非零矩阵(零化数恰 1;已在 (6,5),
(8,4),(6,6) 逐点数值确认)。故

    **dim ker M(Lq,Lr) = 2 + (Lq+Lr+g−3) = Lq+Lr+gcd(Lq,Lr)−1。** □

**定理 3(核的显式结构——三族条带模)**:核由三族模直和生成(数值
逐族验证到机器精度):

- 行模(q 向不变,t 只依赖 r):t_A(r) = ψ − g(r−1),t_B(r) = g(r),
  g 为任意 Lr 周期函数、ψ 任意标量 ⟹ 维数 Lr+1;
- 列模(r 向不变):t_A(q) = σ − G(q),t_B(q) = G(q) ⟹ 维数 Lq+1;
- 对角模(沿反对角线 s = q+r 调制,二部交错):t_A = c·ω^s,
  t_B = −c·ω^s,ω = e^{2πi/g}:g−1 维。

两两交叠仅 in {常向量, 二部交错向量}(2 维)⟹ 总维数
(Lr+1)+(Lq+1)+(g−1)−2 = Lq+Lr+g−1,与定理 2 互为独立证明。
(注:对角模在 g ≥ 3 时非实数值,整数字母表解一般取不到它——精确
计数比边界更小,见 §4。)

**推论(主定理)**:解集 = ker M ∩ A^(2F);消元取 dim 个自由坐标 ⟹

> Ω ≤ k^{Lq+Lr+gcd(Lq,Lr)−1},
> s = ln Ω/F ≤ (Lq+Lr+g−1)·ln k/(Lq·Lr) → 0(任意趋限方式)。

**机制定性(与方格对照)**:同为 z=4 临界、净自由度 0、零化数同为
O(L) 亚广延,但机制不同型——方格是**传播型**(对边相等沿行/列链式
恒等传播,解空间被逐行/列参数化,Ω = k^{Lx+Ly} 饱和边界);kagome 是
**谱型**(符号行列式非零多项式,零点集 = k 环面上三根线,网格只命中
O(L) 个动量;解空间 = 三族条带模的直和,带二部交错扭转,Ω 一般
严格小于边界)。**零熵机制三分(2026-09-27 按 3.4.6.4 反例修订)**:
传播型(方格、Penrose 条带)/ 谱型(kagome,孤立零线)/ **flat-band 型**
(蜂巢非方 3×2 处处亏秩;**3.4.6.4 是方阵系统却带二维零点集 ⟹ 广延
零化数**——旧"方型 ⟹ 零点集孤立线 ⟹ 亚广延"二分不成立,见 z4_3464/)。
统一语法:约束矩阵符号行列式的零点集结构决定零化数增长率。

**边界口径**:开边界拼块的零化数 = O(周长)(数值:条带 s(W,L) 随
c(W+L)/(WL) 衰减,见 §4;定理严格口径为环面)。

## 3. 分类学角色(四图分类闭合)

| 图 | z | 每面净自由度 | 机制 | 零化数(环面) | s(∞) |
|---|---|---|---|---|---|
| 蜂巢 | 3 | +1 | 加和型,符号处处亏秩(flat) | F+2(广延) | **>0(数值;单面型内唯一;全有理混合族另有正熵反例 3.4.6.4,z4_3464/)** |
| 方格 | 4 | 0 | 传播型(对边相等链) | 2L(O(L)) | **=0(SQ 定理,Ω 精确)** |
| **kagome** | 4 | 0 | **谱型(符号零线)** | **Lq+Lr+g−1(O(L),精确)** | **=0(本定理)** |
| 三角格 | 6 | −2 | 过定,等边绑定 | O(1) | **=0(TRI 定理,Ω 精确)** |

至此**平面四图(单面型)分类闭合**:z=3 蜂巢是 {3,4,6} 价单面型
等角网络中唯一的正熵载体;z=4 两支(传播型/谱型)与 z=6 过定端全部
零熵。论文 1 的分类骨架就此完整。
限定修订(2026-09-27):混合面族两支命运已分——**含无理面型的劈行族
4.8.8/3.12.12 零熵已证**;**全有理面型混合族开放,且 3.4.6.4 已证
dim = F_hex+3 广延(正熵强证据,z4_3464/)**。"唯一载体"措辞必须带
"单面型"限定并显式引 3.4.6.4 反例。

## 4. 数值验证(`KAG_数值验证.py` → `KAG_熵密度验证.csv`)

四层独立验证,**全部通过**(脚本断言 ALL CHECKS PASSED):

1. **暴力枚举锚点**(不预设结构):环面 t 全赋值 k^{2F} 逐面过滤,
   与核枚举(核基通分 + 自由坐标枚举)逐位一致;开边界拼块上条带
   TM(前沿 DP)与核枚举逐位一致((2,2)…(4,3) 六例)。
2. **零化数表**:21 个尺寸(含 g = 1,2,3,4,6 的 gcd 结构;小矩阵
   Fraction 精确、大矩阵大素数域)对公式 Lq+Lr+g−1 **零违规**;
   符号行列式对因式分解的连续 k 残差 < 1e−14;零点网格计数与逐点
   2×2 零化数(原点 2、其余 1)在 (6,5)/(8,4)/(6,6) 确认。
3. **精确 Ω 系列**(核枚举):Ω ≤ k^dim 逐点成立且普遍严格小于
   (如 (4,4) k=3:Ω = 3283 vs 边界 3¹¹ = 177147;s 从 (2,2) 的 1.19
   降至 (6,4) 的 0.33;k=2 系列到 (10,5):s = 0.14)。
4. **条带系列**(开边界 TM,状态合并版):s(W,L) 按 c(W+L)/(WL)
   衰减——开边界零化数 O(周长) 的直接表现,与定理自洽。

## 5. 英文 SI 成文草稿(SI-ready)

> **Theorem (zero metric entropy on the kagome lattice).** Let G be the
> Lq×Lr kagome graph on a torus (Lq, Lr ≥ 2) — the medial graph of the
> honeycomb: vertices of valence 4, faces equilateral triangles and
> hexagons in ratio 2:1. The isogonal condition fixes all angles (60° on
> triangles, 120° on hexagons) and every edge length is drawn from a
> finite alphabet of size k. Then the number Ω of valid edge-length
> assignments obeys Ω ≤ k^{Lq+Lr+gcd(Lq,Lr)−1}, and the entropy per face
> s = ln Ω/(Lq Lr) → 0.
>
> **Proof.** (i) *Reduction.* Each triangle has its three sides along the
> three fixed direction families meeting at 60°; closure forces the three
> lengths equal, so every triangle carries a single variable t (one per
> honeycomb vertex, 2F in total). Each hexagon then imposes two
> independent pair-sum equations on the six surrounding t's — a square
> 2F×2F linear system M. (ii) *Bloch decomposition.* M commutes with
> torus translations, so ker M = ⊕_k ker M̂(k) with the 2×2 symbol
>
>     M̂ = [[−1 − a⁻¹b + 2a⁻¹, −2 + a⁻¹ + b⁻¹],
>          [ 2 − a⁻¹b − a⁻¹,  1 − 2a⁻¹ + b⁻¹]],  a = e^{ik₁}, b = e^{ik₂},
>
> whose determinant factors exactly as det M̂ = 3(a−1)(b−1)(a−b)/a²b.
> The zero set is the union of three wrapped lines {k₁=0}, {k₂=0},
> {k₁=k₂}; the momentum grid meets it in Lq+Lr+gcd(Lq,Lr)−2 points, with
> nullity 2 at the origin (the uniform and the staggered mode) and 1
> elsewhere — hence dim ker M = Lq+Lr+gcd(Lq,Lr)−1. (iii) *Counting.*
> Gaussian elimination parametrizes the solution set by dim ker M free
> coordinates, each ranging over the alphabet: Ω ≤ k^{dim ker M}. □
>
> The residual modes are three stripe families (row-, column- and
> diagonal-modulated, the latter with a staggered sign), explicit in the
> SI: subextensive freedom, zero entropy density. Mechanistically the
> kagome branch is *spectral* (isolated zero lines of a nonvanishing
> symbol determinant), to be contrasted with the *propagational* square
> lattice (chained equalities); both are subextensive, while the
> honeycomb's symbol is rank-deficient everywhere (extensive freedom,
> positive entropy) and the triangular lattice is overdetermined
> (rigid). The four-graph classification is thereby closed: among
> single-face-type isogonal planar networks of valence {3,4,6}, the
> honeycomb is the only metric-random-tiling carrier.

## 6. 建模教训(供后续图类复用)

1. **小环面平行边**:Lq 或 Lr = 1 时六边形与自身共边,端点对去重会
   错误合并(与三角格同款);库内 assert Lq, Lr ≥ 2 兜底。
2. **轨道键 vs 槽位几何**:环面上顶点身份用平面坐标对周期格的轨道
   规范化(词典序最小表示);但**几何量(方向、闭合系数)必须用角槽
   的精确平面坐标**计算,不能用轨道代表位置(相差格矢,方向会错)。
3. **面环向取向**:约束行方向必须沿面边界的 CCW 环向取向;按"键对
   排序号"定向会与环向脱钩,60°/120° 断言即报错。
4. **TM 状态合并**:前沿 DP 向量化后,不同新角组合落在相同 after
   槽位上的行必须按键合并(否则状态数组按路径数膨胀,内存爆炸);
   合并后峰值 = 真状态数 = 3^切宽((2,x)=81、(3,x)=633、(4,x)=4143),
   (4,24) 条带秒级。

## 7. 文件与状态

- 本文档 + `kagome_lib.py`(模型库:环面/拼块构建、约束矩阵、精确
  线性代数、核枚举、条带 TM)+ `KAG_数值验证.py` + `KAG_熵密度验证.csv`
  (本目录)。
- 无外部文献依赖;纯框架内定理。待办:并入论文 1 四图分类节 SI
  (第 5 节直接取用);四图分类表已同步本目录与数据地图 README。
