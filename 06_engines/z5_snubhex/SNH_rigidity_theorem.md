# SNH — (3,3,3,3,6) snub hexagonal 刚性:dim_ℝ = dim_ℚ = 1,s(∞) = 0

> 2026-10-01 归档整合(第七十五轮)。定理与数值为 2026-09-27 第二十一轮已定案结果
> (回归 32/32,`z4_3464/RTH_数值验证.py` 第七部分);本文从 `z4_3464/README.md`
> §snub-hexagonal 与 `snub_lib.py` 文档串收拢为单系统档案。

## 定理

**(3,3,3,3,6) snub hexagonal 等角密铺(全有理面族 {3,6},z=5):**

$$\dim_{\mathbb{R}} = \dim_{\mathbb{Q}} = 1 \ (\text{精确},\ k=1..5 \ \text{全验证}) \ \Longrightarrow\ \text{刚性},\ \Omega \le k,\ s(\infty) = 0$$

唯一核模式 = 全局标度(所有边同值)。

## 机制:共边三角形直接锁死

- 每三角形三边等值(等角三角形 60-60-60 ⟹ 等边,闭合恒等式);
- 共边两三角形:共享边同时是两者的等值边 ⟹ 两三角形同值;
- 三角形共边邻接图**连通** ⟹ 全部三角形边 = 单一标度;
- 六边形每边皆邻三角形 ⟹ 全部边同值。
- 与 3.4.6.4 的三角锁死同族但**更强**:那里经方形间接锁(锁死定理 ⟹ 核投影秩 1),
  这里共边直接锁(一步传播)。
- Niven:面 ∈ {3,6} 全有理 ⟹ 无劈行,dim_ℚ = dim_ℝ(有理性不是这里的机制,锁死才是)。

## 构造(三角格完全码 / perfect 1-error-correcting code)

- 三角格 a₁=(1,0), a₂=(1/2,√3/2);码 = ker φ,φ(i,j) = (i+3j) mod 7
  (六个邻居方向映到全部 6 个非零陪集 ⟹ 完全码;码格 Λ_c = ⟨(−3,1),(7,0)⟩,det 7,密度 1/7);
- 中心 = 码点(删除),辐条 = 中心-邻居边(删除);
- 剩余面 = 每中心一个六边形(绕中心 6 邻)+ 全部三顶点非中心的单元三角形;
  顶点形 = 4 三角形 + 1 六边形 = (3,3,3,3,6);
- 环面 = k·Λ_c(与码格相容):格点 7k²,中心 k²,六边形 k²,三角形 8k²,顶点 6k²,
  边 15k²,全顶点度 5 ✓。

## 数值验证

- `snub_lib.py` 内建自检链:完全码性 / Euler 与面计数 / 全 1 闭合 / 顶点形环向归一 = (3,3,3,3,6)
  / dim_ℝ = dim_ℚ(k=1..5);
- 回归:`z4_3464/RTH_数值验证.py` 第七部分 32/32 PASS(2026-09-27);
- 本目录 `SNH_数值验证.py` = 独立重跑入口(本地 snub_lib 副本,k=1..5)。
- 实现注记:边键 = 倍中点模 **2×环面格** 约化(半格点同一性;误用格点约化会把平行边中点
  错误合并——小环面平行边坑的变体)。

## 分类后果

全有理顶点型族闭合(2026-09-27,z=5 三循环序结局齐):
(3,3,3,4,4) 亚广延 2L+1 / **(3,3,3,3,6) 刚性 dim=1** / (3,3,4,3,4) O(1) 双参数(dim=2,2026-09-28 补,
见 `z5_snubsquare/`)。蜂巢之外全部零熵(3.4.6.4 的 ℤ 层以条带 TM 定案为准)。

## 英文 SI 草稿(Draft;供 S8 取用)

> **Theorem (rigidity of the snub-hexagonal tiling).** In any equiangular realization of the
> (3,3,3,3,6) tiling, every triangle is equilateral (60-60-60 closure), and two triangles sharing
> an edge are forced to the same scale; the edge-adjacency graph of triangles is connected, so all
> triangle edges share one value, and every hexagon edge bounds a triangle — one global scale total:
> dim = 1 exactly, Ω ≤ k, s = 0. Since all faces have p ∈ {3,6}, the constraint rows are purely
> rational (no row splitting): dim_ℚ = dim_ℝ. Torus counts with the perfect-code construction
> (kernel of φ(i,j) = (i+3j) mod 7 on the triangular lattice) verify dim = 1 for k = 1…5.

## 核实状态

- 定理(dim=1)与构造:已验(2026-09-27 回归 32/32;本目录独立重跑入口可复现)。
- 机制论证(共边锁死 + 连通性):已验(构造的自检链含全 1 闭合与顶点形断言)。
- 经典对应(3.3.3.3.6 = snub hexagonal / Wythoff s{6,3}):教科书结论,引用前过库。

## 文件

- `SNH_刚性定理.md` — 本文。
- `snub_lib.py` — 构造与零化数(自 z4_3464/ 归档副本;`z4_3464/` 保留原副本供联合回归套件使用,
  两份内容一致。注:副本仍 import `RTH_数值验证` 的 ℚ(√d) 算术机器——运行时需 z4_3464 在 path,
  `SNH_数值验证.py` 已内置该路径)。
- `SNH_数值验证.py` — 独立验证入口(k=1..5)。
- 演示图:无(dim = 1 = 全局标度,解空间无随机性可言;README 已注明)。
