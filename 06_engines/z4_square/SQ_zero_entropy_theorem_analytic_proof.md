# SQ — z=4 方格:等角度量无序的零熵支(解析定理 + 数值验证)

日期:2026-09-26。状态:**定理证明完成,数值验证通过,可写入论文 SI。**

**结论先行**:等角(90°)方格网格上不存在度量 random tiling——矩形闭合条件
("对边相等")沿列/行链式传播,把 E = 2·Lx·Ly 条边的自由度压到恰好
Lx+Ly 个(O(√N),亚广延)。周期边界下解空间**精确**为

> **Ω = k^(Lx+Ly)**(k = 边长字母表大小;项目字母表 {4,5,6} 时 Ω = 3^(Lx+Ly))

每面熵密度 s = (Lx+Ly)·ln k/(Lx·Ly) → 0。这是**零熵的严格定理**,不是数值外推。

---

## 1. 设定与框架约定

与蜂巢框架(AGENTS.md 概念约定)完全同构,仅换图:

- **组合拓扑锁定**:连通 patch 取规则方格图,Lx×Ly 个四边形面(环面为主口径);
  顶点 4 价,4×90° = 360°,无五/七边形缺陷。
- **角刚边柔**:内角全部钉在 90°;每条边一个长度,取自离散字母表 A(|A| = k,
  项目口径 k = 3,边色/边长 4/5/6)。
- **闭合条件**:每个四边形面几何闭合。等角四边形 = 矩形 ⟹ 闭合条件退化为
  **对边相等**:上 = 下、左 = 右(两条等式/面)。

环面记账:V = Lx·Ly,E = 2·Lx·Ly,F = Lx·Ly,满足 Euler V−E+F = 0。
要求 Lx, Ly ≥ 2(=1 时面与自身共边,退化)。
实现注(2026-09-27 审核后补):小环面 L=2 有平行边,边身份必须按
(方向族,行/列号)显式编号,禁止按端点对去重(与 TRI/KAG 同类教训)。

## 2. 定理与证明

**定理 1(传播)**:任何合法构型中,第 i 列的全部横边同长(记 h_i),
第 j 行的全部竖边同长(记 v_j)。

*证明*:面 (i,j) 的矩形条件给 top(i,j) = bottom(i,j),而 top(i,j) 正是
面 (i,j+1) 的 bottom。沿 j 逐面链接(环面上自动回绕),第 i 列所有横边
属于同一条等式链 ⟹ 全等于 h_i。竖边对称:面 (i,j) 的 left = right,
left(i,j) 即面 (i−1,j) 的 right,沿 i 链接 ⟹ 第 j 行所有竖边 = v_j。□

**定理 2(实现性,双向)**:反之,任意 (h_1..h_Lx; v_1..v_Ly) ∈ A^(Lx+Ly)
给出唯一合法构型。

*证明*:顶点坐标取累加 x_i = Σ_{i'≤i} h_{i'}、y_j = Σ_{j'≤j} v_{j'};
面 (i,j) 即矩形 [x_i, x_{i+1}]×[y_j, y_{j+1}],四角均为 90°,对边相等
自动满足;周期闭合自动成立(两个周期分别为 Σh_i 与 Σv_j,对所有行/列
相同)。映射 (h,v) ↔ 构型是双射(边长赋值本身即 (h,v))。□

**推论(主定理)**:Ω = k^(Lx+Ly) 精确;

> s(Lx,Ly) = ln Ω/F = (Lx+Ly)·ln k/(Lx·Ly) → 0(Lx,Ly → ∭,任意方式)。

**自由度记账**(与蜂巢对照):边变量 E = 2LxLy,矩形等式 2F = 2LxLy 条,
秩 = 2LxLy − (Lx+Ly),解空间维数恰 = Lx+Ly = O(√N),**亚广延** ⟹ 熵密度
为零。机制定性:**传播型**——等式沿整行/整列直线路径全局传播,一行/列
只剩一个可自由选的值。

**边界口径**:开矩形 patch(自由边界)参数化完全相同 ⟹ 同样 Ω = k^(Lx+Ly);
钉住一条完整外边界线(或任一条横边+其端点竖边)则进一步锁死若干 (h,v),
Ω 只会更小 ⟹ s = 0 与边界条件无关。

**附注(定理范围,写作时如实声明)**:定理对象是规则方格图(框架锁定组合
拓扑,与蜂巢同一约定)。一般四边形剖分中"对边相等"的传播类 = 直进
(Petrie)路径,其长度可以短于网格跨度,类数可恢复广延——超出本框架,
不属于定理范围。三角格的姊妹定理(见 `z6_triangular/`)则**不依赖**
组合拓扑细节,两定理的这一差异建议在 SI 各写一句。

## 3. 分类学角色(四图分类中的位置)

| 图 | z | 每面净自由度 | 机制 | s(∞) |
|---|---|---|---|---|
| 蜂巢 | 3 | +1(孔心游动) | 加和型闭合,解空间广延 | >0(单面型内唯一;混合族见 z4_3464/) |
| **方格** | 4 | 0(残余 O(L) 行列模) | **传播型**(对边相等沿直线链) | **= 0(本定理,精确)** |
| kagome | 4 | 0 | 加和型(t 加和约束) | 待判定(数值在跑,两头皆定理级) |
| 三角格 | 6 | −2(过定) | 等边绑定,全刚性 | = 0(姊妹定理,精确) |

要点:方格与 kagome 同价(z=4,临界)而约束型不同——方格是"传播型零熵",
给 kagome 的数值判定提供同价对照;方格 s=0 但**非刚性**(残存 O(√N) 个
行/列长波模,类比 phason 的有限版本),这是"零熵 ≠ 零模"的最干净示例。

## 4. 数值验证(`SQ_数值验证.py` → `SQ_熵密度验证.csv`)

三层独立验证,**全部通过**(脚本断言 ALL CHECKS PASSED):

1. **暴力枚举锚点**(不预设结构,对全部 k^E 边赋值逐个过滤矩形约束):

| 环面 | k | E | 枚举 Ω | 公式 k^(Lx+Ly) |
|---|---|---|---|---|
| 2×2 | 3 | 8 | 81 | 81 ✓ |
| 2×3 | 3 | 12 | 243 | 243 ✓ |
| 3×2 | 3 | 12 | 243 | 243 ✓ |
| 3×3 | 2 | 18 | 64 | 64 ✓ |
| 4×2 | 2 | 16 | 64 | 64 ✓ |
| 2×4 | 2 | 16 | 64 | 64 ✓ |

2. **传播类计数**(union-find,仅用矩形等式):类数 = Lx+Ly 精确成立
   (2×2→4, 3×3→6, 4×4→8, 6×4→10, 8×8→16),且类结构检查通过:
   横边类严格按列分(同列同值、异列异值),竖边类严格按行分。
3. **尺寸系列**(k=3,每面熵):2×2: 1.0986 → 4×4: 0.5493 → 8×8: 0.2747
   → 16×16: 0.1373 → 32×32: 0.0687 → 48×48: 0.0458,按 (Lx+Ly)/LxLy
   精确衰减,无拟合、无外推。

## 5. 英文 SI 成文草稿(SI-ready)

> **Theorem (zero metric entropy on the square lattice).** Let G be the
> Lx×Ly square-lattice graph on a torus (Lx, Ly ≥ 2), with every face
> required to be a rectangle — the equiangular condition fixing all four
> angles at 90° — and every edge length drawn from a finite alphabet A of
> size k. The number of valid edge-length assignments is exactly
> Ω = k^(Lx+Ly), and the entropy per face s = (Lx+Ly) ln k /(Lx Ly)
> tends to zero as Lx, Ly → ∞.
>
> **Proof.** (i) *Propagation.* The rectangle condition on face (i, j)
> forces top = bottom and left = right. The top edge of (i, j) is the
> bottom edge of (i, j+1); chaining in j therefore equates all horizontal
> edges of column i to a single length h_i, and symmetrically all vertical
> edges of row j to a single length v_j. (ii) *Realization.* Conversely,
> any (h_1..h_Lx; v_1..v_Ly) ∈ A^(Lx+Ly) defines vertex coordinates by
> cumulative sums; each face is then the rectangle
> [x_i, x_{i+1}] × [y_j, y_{j+1}], and the assignment closes periodically
> with periods Σ h_i and Σ v_j. The map is a bijection on edge-length
> assignments. □
>
> The residual degrees of freedom are the O(Lx+Ly) row and column modes —
> subextensive — so the square lattice carries no metric random tiling:
> its configurational entropy density vanishes exactly. (Scope: the
> square-grid combinatorics. In a general quadrangulation the propagation
> classes are straight-ahead (Petrie) paths and need not be subextensive.)

## 6. 文件与状态

- 本文档 + `SQ_数值验证.py` + `SQ_熵密度验证.csv`(本目录)。
- 无外部文献依赖;纯框架内定理。待办:并入论文 1 四图分类节 SI(第 5 节
  直接取用);可选项:无(不需要图)。
