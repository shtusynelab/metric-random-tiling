# SNB 定理:3.3.4.3.4(snub square)度量刚性——dim = 2 精确,s(∞) = 0

> 2026-09-28 第二十五轮;解析证明(手工验算)+ 数值验收(`_temp/snub_square_rigidity_check.py`,7 尺寸全 PASS)。
> 定位:11 种阿基米德均匀顶点型查漏(AGENTS.md 分类条目此前缺 3.3.4.3.4 与 4.6.12;本文闭合前者)。

## 定理

**3.3.4.3.4 等角(自由边长)密铺的度量解空间:**

$$\dim_{\mathbb{R}} = \dim_{\mathbb{Q}} = 2 \quad (\text{一切合法环面商} \; N,M \text{ 均偶, 精确})$$

核 = 两个类常数向量:菱形对按棋盘格分成两类(i 奇 / i 偶),每类内所有菱形对边长相等;
矩形面成为 a×b 矩形(等角四边形,同 4.8.8 口径)。规则解 = a = b 特例。
**Ω = k² ⟹ s(∞) = 0(定理级)。**

## 引理 1(三角形两两配对)

顶点型 (3,3,4,3,4) 循环邻接多重集 = {3-3, 3-4, 4-3, 3-4, 4-3}:
(a) 每顶点恰一条 T-T 边 ⟹ #TT = V/2;(b) 边型只有 TT、TS,无 S-S。
任一三角形任两边共享顶点;若它有两条 TT 边,该共享顶点便有两条 TT 边,与 (a) 矛盾 ⟹ k_T ≤ 1。
计数:Σ_T k_T = 2·(V/2) = V,而 F₃ = 3V/3 = V ⟹ **全体 k_T = 1**:三角形经 TT 边完美配对成
60/120 菱形(TT 边 = 短对角线,连接两个 120° 角;|短对角线| = 边长,勾股余弦:三角形 v-p-w 在 p 处 60°)。

## 引理 2(归约:解空间 ≅ 菱形对图上的局部常值函数)

两三角形闭合(各秩 2)共享 TT 边 ⟹ 每对菱形 5 条边全等长,记 a_ρ(每对恰一个自由参数)。
由引理 1(b),每条边都是三角形边 ⟹ 矩形四边 = 四个相邻菱形对的 a 值;
矩形闭合(秩 2)⟺ 对边相等 ⟺ **a_ρ1 = a_ρ3, a_ρ2 = a_ρ4(跨矩形相对)**。
故解空间 ≅ {a : 菱形对 → ℝ 满足全部对面相等} = G′ 上的局部常值函数空间,
**dim = G′ 连通分支数 c**(G′:节点 = 菱形对,边 = 跨矩形相对)。

## 引理 3(复合棋盘格;G′ 恰两分支)

复合层(菱形作为整体单元):每菱形边 = 矩形边、每矩形边 = 菱形边,顶点周围 R₁₂₀, R₆₀, S, S 循环。
构造性识别:ℤ² 方格网偶格放矩形、奇格放菱形;菱形按定向 o = i mod 2 切短对角线(120° 角在其两端,
相邻菱形定向由顶点角和 360° 强制交替,等价于菱形邻接图{{对角步}}的二染色,即 i 的奇偶)。
逐顶点核验:2 菱形角(一 120 一 60)+ 2 矩形角 = 120+60+90+90 = 360°,顶点型恰 (3,3,4,3,4) ⟹
这就是 snub square 组合(组合型唯一性 = 经典 Archimedean 唯一性【教科书结论,引用前过库】)。
G′ = 奇格跨格距离-2 图,步 (±2,0),(0,±2) 全部保持 i 奇偶 ⟹ **恰两分支 {i 奇}、{i 偶}**(平面)。
定向图案把平移格限到 ⟨(2,0),(0,2)⟩(单位格平移翻转定向,非对称)⟹ 一切合法环商两类都在 ⟹ c = 2。

## 推论与对照

| 系统 | 顶点型 | dim | 机制 |
|---|---|---|---|
| 拉长三角 | 3.3.3.4.4 | 2L+1(亚广延) | 相邻方块=多米诺,条带滑动 |
| **snub square(本文)** | **3.3.4.3.4** | **2 = O(1)** | TT 配对杀死条带;只剩双色双尺度 |
| snub hexagonal | 3.3.3.3.6 | 1 | 三角格完全码 |

关键对照:3.3.3.4.4 与 3.3.4.3.4 同字母表 {3,3,3,4,4}、同 z=5、naive 净自由度同为 −V/2,
仅因循环序不同(两个 4 相邻 vs 不相邻)结局差一个数量级——循环序决定命运,与 5-5 motif 敏感性同构。
naive 记账再次失效的定量注脚:rows − rank = **V/2 + 2**(数值逐位确认),与 3.12.12(naive −5V/6 而 ℝ 广延)同教训。

## 数值验收(`_temp/snub_square_rigidity_check.py`;K=ℚ(√3) 精确消元,rank_ℝ = rank_K 因 K⊂ℝ 子域)

| N | M | V | E | rank | dim | 核基 = vI, vII | 判定 |
|---|---|---|---|---|---|---|---|
| 2 | 2 | 4 | 10 | 8 | 2 | ✓ | PASS |
| 2 | 4 | 8 | 20 | 18 | 2 | ✓ | PASS |
| 4 | 4 | 16 | 40 | 38 | 2 | ✓ | PASS |
| 4 | 6 | 24 | 60 | 58 | 2 | ✓ | PASS |
| 6 | 6 | 36 | 90 | 88 | 2 | ✓ | PASS |
| 4 | 8 | 32 | 80 | 78 | 2 | ✓ | PASS |
| 6 | 8 | 48 | 120 | 118 | 2 | ✓ | PASS |

结构断言全过:Euler、全顶点度 5、#TT = V/2、每三角形恰 1 TT 边、TT/TS 边型、方向指派双面一致(差 6)。

## 物理判读(单独成节,供审查)

1. snub square 无度量熵:二元/多元合金化若落到此拓扑,构型熵项为零,单相判据全押在混合焓上
   (与三角格、kagome 同侧,与蜂巢、3.4.6.4 反侧)。
2. "每六边形一模"语法的反例面:本系统无六边形,刚性来自**三角形配对的完美匹配**——
   三角锁死(3.4.6.4 的全局单一值)在这里降级为"每菱形对一模",两个残余参数是染色类的,
   不是几何滑移。分类表补全后,z=5 行三种循环序 = 三种结局(亚广延 / O(1) 双参数 / O(1) 单参数),
   "临界维数"叙事可在 R2 分类节直接引用。
3. 手性无关性:证明全程未用手性(镜像对的 a-系统相同),两枚 snub 变体同判。

## 英文 SI 草稿(Draft)

**Theorem (metric rigidity of the snub-square tiling).** In any edge-to-edge tiling whose vertices
all have cyclic type (3,3,4,3,4), every triangle shares exactly one edge with another triangle
(counting: each vertex carries exactly one T–T edge, so #TT = V/2, while a triangle cannot carry
two T–T edges without forcing two such edges at a shared vertex; Σk = 2·#TT = V = F₃ forces k ≡ 1).
Triangles therefore pair into 60/120 rhombi whose five edges share a common length a_ρ. Every edge
bounds a triangle, so each rectangle side inherits the a-value of its adjacent pair, and rectangle
closure is equivalent to equality of opposite pairs: a_ρ1 = a_ρ3, a_ρ2 = a_ρ4. The metric solution
space is thus the space of locally constant functions on the "opposition graph" G′ of rhombi; its
dimension equals the number of components of G′. The composite (rhombus + rectangle) tiling is a
checkerboard square grid with rhombi cut along orientation-alternating short diagonals; G′ is the
distance-two graph of the checkerboard and has exactly two components {i odd}, {i even}. Hence
dim_ℝ = dim_ℚ = 2 on every admissible torus (N, M even), with kernel spanned by the two
class-constant vectors: rectangles become a×b rectangles and each rhombus class carries one scale.
The entropy density vanishes: Ω = k². Exact rational arithmetic in ℚ(√3) on tori up to 48 vertices
confirms dim = 2 and the stated kernel basis in all cases.

## 实现层观察:周期格与 holonomy(2026-10-01,演示图作图时发现;数值实证,正式小命题待写)

核元素 (a,b) 的**几何实现不受环面限制**:任一合法环面(N,M 偶)上,把平面 (a,b)-铺砌按
4 细胞平移作商即得平坦环面实现——(a,b) 铺砌仍以 2 细胞为周期(方向图案 metric 无关,
长度按类保持),周期格 U(a,b) = h₁/2、V(a,b) = h₂/2 随 (a,b) 连续变形。
4×4 环面实测(dir_of 规范下,`_temp/metric_demos/snub_holonomy_check.py`):

- 生成元位移 h₁(vI) = (√3, −1)、h₁(vII) = (2, 0)、h₂(vI) = (0, 2)、h₂(vII) = (1, √3);
  h 线性于 (a,b);纬度圈行无关(圈同调检验 ✓,面闭合的推论)。
- 布局 BFS 的"clash 向量"逐位等于 h₂:(1, 3.732)/(0.75, 3.299)/(1.33, 4.304)
  对 a/b = 1/0.75/1.33——即不同 lift 之差 = 平移周期,**是布局记账问题,不是度量不一致**
  (规则解 a=b=1 同样 clash,排除度量故障解读)。

判据口径(候选小命题):环面上长度赋值可实现为平坦环面度量 ⟺ 面闭合 + 生成元位移 h₁、h₂
构成秩 2 格( immersion 条件);snub square 核内全元素通过 ⟹ dim = 2 的几何实现在环面上无
holonomy 障碍。注意与熵口径分层:Ω = k² 数的是面闭合赋值(组合构型),实现层是它的几何
具体化,两者在此系统恰好一致(核全体可实现)。

## 核实状态

- 引理 1–3 与定理:本会话手工推导,逐步验算;数值验收 7/7 PASS(本机,精确算术)。
- "组合型唯一性 / Archimedean 唯一性"、"21 个角度合法顶点型"的教科书对应:凭记忆与推导一致,成文引用前过库。
- 复合棋盘构造的定向图案唯一性(二染色):已在本会话手工验证(菱形邻接图二部性 ⟹ 恰两个全局选择)。
