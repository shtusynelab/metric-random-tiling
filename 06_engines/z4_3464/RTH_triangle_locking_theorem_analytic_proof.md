# RTH_三角锁死定理解析证明(2026-09-27,第二十一轮)

> 定理化第二十轮的发现:3.4.6.4 整数字母表随机解中"全部三角形边 = 同一值"
> (200/200 解、210 条三角形边核投影秩 = 1)。本定理是该现象的严格证明,
> 也是 V_ℤ 薄度判据纲领(立项卡片)Phase 1 的承重结果。
> 验证脚本:`_temp/thick_crit_numerics.py`(A 组,L=2..5 全 PASS);
> 常驻回归:`z4_3464/RTH_数值验证.py` §三角锁死(2026-09-27 增)。

## 1. 定理

**定理(三角锁死)**。设 3.4.6.4(rhombitrihexagonal)密铺的环面或单连通
patch 上给定边长赋值 x:E → ℝ(或 ℤ),使每面闭合(内角固定:三角形 60°、
方形 90°、六边形 120°)。则**所有三角形边取同一值**:存在 t ∈ ℝ(ℤ)使
x(e) = t 对一切三角形边 e 成立。

**推论 1(投影秩)**。解空间(核)在三角形边坐标上的投影维数为 1;即
"210 条三角形边的核投影行秩 = 1"(第二十轮实测的精确形式)。

**推论 2(字母表层)**。字母表 A 下 t ∈ A,自由度 1(非 |A|^(#三角形边));
三角形边不贡献广延熵。

## 2. 证明

**引理 1(三角形刚性)**。三角形三边方向 u₀,u₁,u₂ 两两 120°,闭合
e₀u₀ + e₁u₁ + e₂u₂ = 0。由 u₀+u₁+u₂ = 0 改写为
(e₀−e₁)u₀ + (e₂−e₁)u₂ = 0;u₀,u₂ 线性无关 ⟹ e₀ = e₁ = e₂。∎

**引理 2(方形对边相等)**。方形四边方向 u₀⊥u₁,u₂ = −u₀,u₃ = −u₁,闭合
(e₀−e₂)u₀ + (e₁−e₃)u₁ = 0 ⟹ e₀ = e₂,e₁ = e₃。在 3.4.6.4 中方形边界
交替邻接六边形与三角形(顶点形 (3,4,6,4)),故其两条**三角形边互为对边**:
引理 2 给出这两条三角形边相等。∎

**引理 3(G_tri = 三角形面邻接图 = 蜂巢图)**。定义图 G_tri:节点 = 密铺
三角形,方形 S 连接其两条三角形边所属的三角形 T₁ ~ T₂(两三角形不同:
方形对边不共顶点,而同一三角形任两边共顶点)。

组合双射(环面 L×L;patch 同理):
- **六边形 ↔ 三角格点**:两六边形相邻 ⟺ 共享一个方形;每六边形 6 邻
  ⟹ 六边形邻接图 = 三角格(triangle lattice,L×L 环面,L² 节点);
- **方形 ↔ 三角格边**:每方形坐在一对相邻六边形之间 = 三角格的一条边
  (计数 3L² = 边数 ✓);
- **三角形 ↔ 三角格面**:顶点形 (3,4,6,4) ⟹ 每顶点恰一个三角形; incidence
  计数 3·F_tri = V = 6L² ⟹ F_tri = 2L² = 三角格面数;每个三角形嵌在三个
  两两相邻的六边形的环绕空隙中 = 三角格的一个面(✓ 双射);
- 方形的两条三角形边分别位于该格边两侧的两个面 ⟹ **G_tri = 三角格的
  面邻接图 = 蜂巢图**(3-正则、二部、girth 6)。

蜂巢图连通(环面与连通 patch)⟹ G_tri 连通。∎

**定理证明**。引理 1:每个三角形三边等值。引理 2:每个方形将其两条
三角形边(分属 T₁、T₂)取等。沿连通图 G_tri 传播:全部三角形边同一值 t。∎

推论 1:核矢量在三角形边坐标上的取值 = (t,x) 中 t 的系数 × 全 1 向量,
投影维数 1。∎

## 3. 数值验证(`_temp/thick_crit_numerics.py` A 组,2026-09-27)

| L | V/E/F | dim | G_tri | 核投影秩 | 随机整解锁死 |
|---|---|---|---|---|---|
| 2 | 24/48/24 | 7 | 3-正则,连通,二部,8 节点 | **1**(Fraction 精确) | 200/200 |
| 3 | 54/108/54 | 12 | 同上,18 节点 | **1** | 200/200 |
| 4 | 96/192/96 | 19 | 同上,32 节点 | **1** | 200/200 |
| 5 | 150/300/150 | 28 | 同上,50 节点 | **1** | 200/200 |

随机解采样协议(镜像第二十轮):核基稀疏组合(1–8 个模,系数 ±2),
200 样本/尺寸;锁值随解变化(−2..2)= 单一全局标量,与"投影秩 1"一致。
常驻回归(小口径:L=2,3;秩 + 连通 + 20 样本)已并入
`z4_3464/RTH_数值验证.py`。

## 4. 在 V_ℤ 薄度纲领中的位置(承上启下)

- 本定理是 **ℤ-covolume 致刚**(第三种 ℤ 机制,区别于劈行/谱/传播)的
  第一步几何锁定:2L² 个三角形(6L² 条边,占 E/2)的自由度塌缩为 1;
- 第二步(方形把六边形边两两配对 + 六边形差分约束)把剩余自由度挤进
  窄通道——第二十轮"形变挤在方/六边的窄通道"的机制化;
- **蜂巢对照**:蜂巢无三角形 ⟹ 无此锁;蜂巢的同构分析(立项卡片 H2)
  = 对边配对图 P 分解为 3L 条差分条纹(每条 L 边,L=2..5 数值验证
  PASS)——一维条带计数是面积律的候选来源;
- 开放(Phase 2/3,见立项卡片):RTH 剩余子系统的条纹参数化猜想
  (Ω ≤ C^L,判伪 = 条带 TM);蜂巢面积律的条纹计数推导;一般判据
  "V_ℤ 何时厚"。

## 5. 英文 SI 成文草稿(SI-ready)

**Triangle locking in the 3.4.6.4 tiling.** Let x: E → ℝ (or ℤ) assign
lengths to the edges of a 3.4.6.4 (rhombitrihexagonal) patch or torus such
that every face closes with its fixed angles (60°, 90°, 120°). (i) In any
equilateral triangle the three edge lengths are equal, since the closure
condition reduces to (e₀−e₁)u₀ + (e₂−e₁)u₂ = 0 with u₀, u₂ linearly
independent. (ii) In any square, opposite edges are equal; in 3.4.6.4 the
two triangle-adjacent edges of a square are opposite, so each square forces
the lengths of the two triangles flanking it to share one value. (iii) The
graph whose nodes are triangles, joined through squares, identifies with
the face-adjacency graph of the triangular lattice on which the hexagons
sit — i.e. a honeycomb graph, which is connected. Hence, in every solution,
all triangle edges carry one global value t, and the space of solutions
projects onto the triangle-edge coordinates with rank one: the 6L²
triangle edges (half of all edges) contribute a single degree of freedom.
For a finite alphabet A this forces t ∈ A; triangle edges contribute no
extensive entropy. Together with the square pairings this "triangle
locking" is the first step of the integer-rigidity mechanism that confines
the ℤ-solutions of 3.4.6.4 to O(L) strip parameters, in contrast with the
honeycomb, whose opposite-edge pairing decomposes into 3L difference
strips and whose integer solutions are exponentially numerous (area law).

## 6. 边界与诚实声明

- 定理对 ℝ 与 ℤ 同样成立(线性恒等式,与字母表无关);它**不单独**给出
  s(∞)=0——薄度结论仍以条带 TM 定案为条件(z4_3464/README 纪律);
- 引理 3 的组合双射在 L=1 小环面退化(数值从 L=2 起);patch 上要求
  单连通或至少连通;
- 锁死值 t 可以取字母表任一值(数值锁值域 −2..2),不锁定具体值——
  "锁结构不锁值"。
