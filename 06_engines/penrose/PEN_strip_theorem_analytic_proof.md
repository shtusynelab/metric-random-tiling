# PEN — Penrose 准晶无法引入度量无序:菱形条带定理

日期:2026-09-27。状态:**定理证明完成,数值验证通过,可写入论文 SI。**
用户提议的"Penrose 无法引入度量无序"**已证明为真——但理由不是顶点
连通度大**,见 §3 的修正。

**结论先行**:

> **Ω = k^(条带数) 精确**(任何 edge-to-edge 菱形等角密铺,含 Penrose);
> Penrose 的条带 = 五网格的线(de Bruijn 对偶中条带 ↔ 网格线,一一对应)
> ⟹ 窗口半径 R 的拼块条带数 = **O(R)**,面数 = O(R²)
> ⟹ **s(∞) = 0**。
> 数值:R=2→13,条带/面比 0.483 → 0.058(→0),最长条带 11 → 96(宏长)。

机制定性:**传播型**——与方格同一族(对边相等沿直进路径全局传播),
不是谱型(kagome)、不是过定型(三角格)。**非周期长程有序不购买任何
度量熵**:Penrose 的准晶序住在组合层(拼接规则),度量层被条带参数化
压到 O(R) 个自由度。

---

## 1. 设定与"价"的修正

- 对象:固定的 Penrose 密铺组合(或任意 edge-to-edge 菱形密铺),度量
  无序 = 边长取自离散字母表、角度保持(瘦菱形 36/144°,肥菱形 72/108°)。
- **价修正(用户直觉"顶点连通度较大"不成立)**:菱形密铺 E = 2F
  (每面 4 边、每边 2 面),Euler 给 V = F ⟹ **平均价 z̄ = 2E/V = 4**
  ——与方格/kagome 同为临界预算,并非"价大过定"。Penrose 顶点价
  实为 3/4/5 混合(平均值恒 4)。正确的零熵机制是对边相等传播(§2),
  价只是预算账本的中性角色。
- 与四图分类的关系:预算口径 net = E−2F = 0(临界),但菱形面的约束
  在**实数层面就是纯等式**(下引理),比 kagome 的加和型更强 ⟹ 直接
  归入传播族。

## 2. 定理与证明

**引理(平行四边形绑定)**:面为四边形,对角相等(α,β,α,β 环向)、
四边方向属固定族(方向 u_α, u_β 不共线)。闭合:
(L₁−L₃)u_α + (L₂−L₄)u_β = 0 ⟹ **L₁ = L₃ 且 L₂ = L₄**(对边相等)。
□(实数层面即等式,无任何无理性前提——与 TSQ 的劈行机制不同源。)

**定理(条带参数化)**:对边相等把边两两配对;配对的传递闭包 = 条带
(直进/蠕虫路径)。解 ⟷ 条带取值:任给每条带一个字母表值,每面得
两对对边值,平行四边形闭合自动满足;反之任何合法解在条带上恒值。
映射是双射,故

> **Ω = k^(条带数)** 精确(任意开拼块/环面)。

**推论(Penrose 零熵)**:de Bruijn 五网格对偶中,族 i 的条带与族 i
的网格线一一对应(菱形跨线配对 = 蠕虫骑线);窗口半径 R 内网格线数
= O(R) ⟹ 条带数 = O(R),面数 = O(R²) ⟹ s = O(R)ln k/O(R²) → 0。
□(对应关系的构造性证明就在生成算法里:`penrose_lib.make_penrose_patch`
按线-交叉枚举,数值上条带/面比实测趋于零,§4。)

**范围附注**:定理对象是**一切**菱形等角密铺:条带数亚广延 ⟹ s=0;
一般(非 Penrose 型)菱形密铺的蠕虫可以短,条带数可广延 ⟹ 预算临界
下重新打开正熵可能——与方格定理的范围附注同款,写作时如实声明。
方格定理(SQ)即本定理的 90° 特例(条带 = 行+列,Ω = k^(Lx+Ly)),
两文档互引。

## 3. 分类学后果(三部曲第三条腿锐化)

| 对象 | 组合序 | 度量熵 | 机制 |
|---|---|---|---|
| 晶体(周期堆垛,如三角格) | 周期 | **0** | 过定/等边绑定 |
| **准晶(Penrose)** | **非周期长程有序** | **0(本定理)** | **传播(条带)** |
| 方格 / kagome(z=4 临界) | 周期 | 0 | 传播 / 谱型 |
| **蜂巢赝晶(z=3)** | 局域规则、长程无序 | **>0(载体之一;全有理混合族另见 3.4.6.4,z4_3464/)** | 加和型广延 |

三部曲的收束句:晶体的周期性与准晶的非周期性都是**组合层**的序,它们
都不给度量层留下熵;度量 random tiling 的熵只住在"局域规则精确 + 长程
无序 + 方向代数无劈行"的蜂巢支。这给论文 1 的密铺词典提供了第三条
可检验预言(局域规则 → 衍射特征 → 熵密度)在准晶行的取值。

## 4. 数值验证(`PEN_数值验证.py` → `PEN_条带验证.csv`)

三层,**ALL CHECKS PASSED**:
1. 结构:面型恰 10 种 {i,j}(i≠j),每面 4 边、对边同族(零违例);
2. 标度(R=2..13):条带/面比 0.483 → 0.058 单调降;最长条带 11 → 96
   (宏长);s = 条带·ln3/面数 0.53 → 0.063(→0);
3. 精确 Ω:小窗(R=1.0/1.2)等式行核枚举 = 3^条带数(531441=3¹²,
   4782969=3¹⁴);微窗 DFS 暴力(2 值)= 2^条带数(16384=2¹⁴)。

生成器:`penrose_lib.make_penrose_patch`——de Bruijn 五网格对偶的组合
构造(交叉↔面、线段↔边、胞↔顶点),正则无理偏移避开三线共点
(注意:偏移勿取 √5 域数值,会与五次几何共振产生退化,首跑即中)。

## 5. 英文 SI 成文草稿(SI-ready)

> **Theorem (no metric disorder on rhombic tilings; Penrose case).**
> Let T be an edge-to-edge tiling whose faces are parallelograms with
> angles fixed by the isogonal condition (e.g. a Penrose tiling), with
> edge lengths from a finite alphabet of size k. In every valid
> assignment opposite sides of each face are equal — writing the closure
> as (L₁−L₃)u_α + (L₂−L₄)u_β = 0 with u_α, u_β independent. The
> opposite-pairing thus propagates from edge to edge through faces; its
> connected components (strands) carry constant values, and every
> strand-value assignment closes every face. Hence Ω = k^{#strands}
> exactly. For the Penrose tiling, de Bruijn's pentagrid duality
> identifies strands with grid lines, of which a window of radius R
> meets O(R); since the number of faces is O(R²), s = ln Ω/ℱ → 0.
>
> Thus aperiodic long-range order — the defining feature of the
> quasicrystalline row of the tiling dictionary — leaves no metric
> entropy: combinatorial order and metric disorder are independent
> resources, and only the honeycomb branch of the classification
> (local rules exact, long-range order absent, rational direction
> algebra) carries positive metric entropy. □
>
> (Scope: all rhombic isogonal tilings. A generic rhombic tiling with
> short worms can have extensive strands; the Penrose claim uses that
> its strands are the macroscopic Ammann worms. Note the average vertex
> valence of any rhombic tiling is z̄ = 4 — the zero-entropy mechanism
> is propagation, not overdetermination.)

## 6. 文件与状态

- 本文档 + `penrose_lib.py`(五网格对偶生成器+条带分析)+
  `PEN_数值验证.py` + `PEN_条带验证.csv`(本目录)。前缀 PEN_。
- 待办:并入论文 1(三部曲准晶行 + 与 SQ 定理互引);de Bruijn 1981
  引用在投稿前按纪律过库核实;可选:有几何坐标后补 FFT 对照
  (Penrose 锐斑无弥散,预期与蜂巢晕对照鲜明)。
