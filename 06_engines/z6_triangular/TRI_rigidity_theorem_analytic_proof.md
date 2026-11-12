# TRI — z=6 三角格:等角度量无序的刚性支(解析定理 + 数值验证)

日期:2026-09-26。状态:**定理证明完成,数值验证通过,可写入论文 SI。**

**结论先行**:等角(60°)三角网格上不存在度量 random tiling——每个三角形的
等边约束(2 条等式)把三边绑死,共享边把绑定传播到全图,**全部边同长**。
解空间只剩整体标度的选择:

> **Ω = k**(k = 字母表大小;环面,任选一条边长 ⟹ 其余全部确定)
> 钉住任一条边(或取 |A|=1)则 **Ω = 1**:唯一解 = 正三角密铺

每面熵密度 s = ln k/F → 0(F = 面数)。这是**过定刚性**,比方格的"亚广延
零熵"更强:连 O(√N) 的残余模都没有,自由度 O(1)。

---

## 1. 设定与框架约定

与蜂巢框架(AGENTS.md 概念约定)同构,仅换图:

- 组合拓扑:规则三角格(环面 L×L 或六角盘半径 R);顶点 6 价,
  6×60° = 360°;面 = 三角形。
- 角刚边柔:内角全部钉在 60°;边长取离散字母表 A(|A| = k = 3,边长 4/5/6)。
- 闭合条件:每个三角形几何闭合。等角三角形 = **等边三角形** ⟹ 闭合退化为
  三边两两相等(2 条独立等式/面)。

记账(环面):V = L²,E = 3L²,F = 2L²,Euler V−E+F = 0;
六角盘半径 R:V = 3R²+3R+1,E = 9R²+3R,F = 6R²。

## 2. 定理与证明

**引理 1(等边绑定)**:每个三角形的三边等长。内角和 180°,三角均 60°
⟹ 等边三角形 ⟹ 三边两两相等。

**引理 2(传播,承重引理)**:任意连通三角剖分(不限于规则三角格)的
三角形邻接图(共享边即相邻)连通。于是等边绑定沿共享边传播:相邻两
三角形共享边 e,两者均等边 ⟹ 两三角形全部六条边长 = |e|;链式推广
⟹ **全图所有边同长**。

**定理(刚性/唯一解)**:合法构型 ↔ 字母表 A 的单个元素 ℓ(全图统一边长);
且任意 ℓ ∈ A 都可实现(即边长 ℓ 的正三角密铺,必然闭合)。故环面
Ω = k;若钉住任一条边的长度,Ω = 1。

**推论**:s = ln k/F → 0;钉边口径下解空间为零,唯一度量 = 正三角密铺。

**自由度记账(过定)**:边变量 E = 3V;等边等式 2F = 4V > 3V = E——
约束数超过变量数 V 之多。等式组相容(正三角密铺是解)但把解压到
1 维(整体标度)。对比:蜂巢每面净自由度 +1(广延 ⟹ 正熵),方格
0(残余 O(√N) 模 ⟹ 零熵非刚性),三角格 **−2(过定 ⟹ 全刚性)**。

**与方格定理的关键差异(SI 值得各写一句)**:本定理**不依赖组合拓扑
细节**——引理 2 对任意连通 6 价三角剖分成立;方格的传播类 = 行/列直线,
在一般四边形剖分中可失效(Petrie 路径变短),定理限于规则方格图。

## 3. 分类学角色(四图分类中的位置)

z=6 是平面等角网络的**刚性端**:过定把度量无序彻底消灭。与之对照,
z=3 蜂巢是唯一已证正熵载体(每孔 +1 自由度),z=4 是临界价
(方格传播型零熵 / kagome 加和型待判定)。四图分类至此两端闭合:
**柔性端(z=3)与刚性端(z=6)均解析,临界端(z=4)一侧解析、
一侧数值在跑**。物理读法:三角格上每个三角形被两侧邻居的等边
条件同时钉死(每条边被两个三角形"上锁"),nature 不给 z=6 留
度量无序的生存空间——这正是已知 random tiling 都需要正熵密度、
而正熵载体落在低价/低价面网络的分类学原因。

## 4. 数值验证(`TRI_数值验证.py` → `TRI_熵密度验证.csv`)

三层独立验证,**全部通过**(脚本断言 ALL CHECKS PASSED;Euler 示性
盘 V−E+F=1、环面 =0 全部内置 assert):

1. **暴力枚举锚点**(k^E 全赋值逐个过滤等边约束):

| patch | V | E | F | 枚举 Ω | 公式 k |
|---|---|---|---|---|---|
| 六角盘 R=1 | 7 | 12 | 6 | 3 | 3 ✓ |
| 环面 L=2 | 4 | 12 | 8 | 3 | 3 ✓ |

2. **类计数 + 承重引理**(union-find 仅用等边等式):11 个 patch
   (盘 R=1..6、环面 L=2..8)上类数全部 = 1(Ω = k),三角形邻接图
   全部连通,零孤立边。注意验证覆盖**盘与环面两种拓扑**,与定理
   "不依赖边界条件/组合细节"一致。
3. **尺寸系列**(k=3,每面熵 s = ln3/F):盘 R=1: 0.1831 → R=4: 0.0114
   → R=8: 0.0029;环面 L=2: 0.1373 → L=6: 0.0153 → L=12: 0.0038,
   按 1/F 精确衰减,无拟合、无外推。

**建模教训(留给后续 kagome 计数,勿踩同坑)**:环面小尺寸下
L=2 出现平行边,边索引必须按 (方向族, q, r) 显式编号——用端点
frozenset 去重会把平行边错误合并(E 12→6,自由度被多删);盘的
down 三角形参数点可以在盘外,扫描域须外扩一圈只测三顶点。
两处 bug 首跑即中,已修复并写入脚本头注释。

## 5. 英文 SI 成文草稿(SI-ready)

> **Theorem (rigidity of the triangular lattice).** Let G be any
> connected 6-valent triangulation — in particular the L×L triangular
> torus or a hexagonal disk — with every face angle fixed to 60° and
> every edge length drawn from a finite alphabet A of size k. Then all
> edges of G share one common length: Ω = k on the torus, Ω = 1 if a
> single edge is pinned, and the entropy per face s = ln k / F → 0.
>
> **Proof.** Each triangle has angle sum 180° with all three angles equal
> to 60°, hence is equilateral: its three edges are equal. Two
> edge-adjacent triangles share an edge, so equilaterality propagates
> across the triangle-adjacency graph, which is connected for any
> connected triangulation; hence all edges are equal. Conversely every
> uniform assignment is realized — by the regular triangular tiling,
> which closes automatically. □
>
> Unlike the square-lattice case, the result is independent of the
> combinatorial details of the mesh: the equilateral constraints are
> overdetermined (2F = 4V independent equalities against E = 3V edge
> variables), leaving no degree of freedom beyond the global scale. The
> triangular lattice thus admits no metric random tiling at all — the
> rigid endpoint of the planar equiangular classification.

## 6. 文件与状态

- 本文档 + `TRI_数值验证.py` + `TRI_熵密度验证.csv`(本目录)。
- 无外部文献依赖;纯框架内定理。待办:并入论文 1 四图分类节 SI(第 5 节
  直接取用);可选项:无(不需要图)。
