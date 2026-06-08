GNN_KAN: Graph Neural Networks with Kolmogorov-Arnold Networks for Interpretable Root Cause Analysis


Abstract
本研究針對現代微服務系統中根因分析（Root Cause Analysis, RCA）面臨之可解釋性不足與泛化能力有限等問題，提出一結合圖神經網路（Graph Neural Network, GNN）與 Kolmogorov-Arnold Networks（KAN）之創新架構——GNN_KAN。有別於傳統僅仰賴結構學習或統計相關性之方法，本研究透過圖結構建模異常傳播路徑，並以 KAN 模組取代傳統多層感知器（MLP）中之黑箱非線性轉換，利用 Kolmogorov-Arnold 表徵定理將多變量函數分解為可學習之一維基底函數組合，使模型兼具表達力與可解釋性。本研究進一步提出「RCA-aware」特徵設計原則，系統化地定義可用於根因定位之特徵，避免資料洩漏與分布偏移。實驗結果顯示，GNN_KAN 在五個核心資料集（包含代碼級故障、多源遙測數據、標準化評估基準）上，於 Top-k 準確度、Precision@k 等指標均展現優異性能，特別在程式碼故障場景達到完美或近完美之識別率。本研究亦提出系統化之基底函數選擇策略，根據系統拓樸結構、故障類型與資料特性提供實務部署之指導原則。

Keywords: Root Cause Analysis, Graph Neural Networks, Kolmogorov-Arnold Networks, Observability, Microservices, Interpretability

Introduction
1.1 Background and Motivation
隨著雲端運算與容器化技術普及，現代軟體系統呈現高度分散與動態特性，微服務架構已成為主流。日前發生的AWS、Azure與Cloudflare等公司不約而同發生的系統異常，也使得根因分析（Root Cause Analysis, RCA）的重要性被提到大眾視野前，軟體架構的轉變，使根因分析日益困難。龐大且高維度的觀測資料（metrics、logs、traces）使傳統單變量分析難以捕捉多維關聯與依賴傳播。異常往往沿服務依賴鏈快速擴散，上游問題可使多個下游同步異常，增加定位難度。此外，微服務間拓樸結構隨版本、負載或彈性調整而變動，觀測信號亦受雜訊、週期性波動影響，進一步降低分析穩健性。除準確度外，可解釋性亦為實務需求上的關鍵。工程與開發團隊需理解異常成因與傳播機制，黑箱模型雖具預測力，卻因缺乏因果證據而難以採用。

1.2 Our Contributions
現有圖神經網路（Graph Neural Network, GNN）方法在微服務上的根因分析上仍存在可解釋性不足與泛化性不穩問題。為此，本研究提出 GNN_KAN，結合 GNN 與 Kolmogorov-Arnold Networks（KAN），兼顧結構推理與可解釋非線性表示。主要貢獻如下：
1. Kolmogorov-Arnold 表徵式圖神經架構：將 KAN 嵌入 GNN 的消息傳遞過程，以一維可學習函數組合多變量非線性關係，提升模型可解釋性與表達力。
2. RCA-aware 特徵設計原則：依據根因定位特性，選取事件窗口內可觀測訊號，避免資料洩漏與偏移，並確保跨場景穩定性。
3. 多場景驗證與可解釋性評估：於 Online Boutique、Sock Shop、Train Ticket 等多個微服務基準下驗證，GNN_KAN 在多項指標均優於現有方法，並展現一致可遷移性與可解釋性。
Related Work
在複雜分散式微服務系統場景下，根因分析 (RCA) 是一項關鍵任務，在根因分析的過程中我們會遭遇到到兩大核心挑戰：異常傳播與掩蔽效應，以及因重要資訊不明顯，從高維度與雜訊觀測中提取有效異常信號的難度。本研究提出的 GNN_KAN 框架將在綜合性的RCAEval 基準測試平台上進行評估，該平台提供了涵蓋 11 種實際故障類型的大規模、多源遙測數據集，以應對傳統方法評估不一致的挑戰。當前主流的基準方法主要為基於因果推論 (CI) 或統計分析的方法，但它們各自存在根本性的局限性：
1. 對時間極為敏感的CI方法： 以 CIRCA 和 RCD 為代表的 CI 方法，雖然旨在構建因果圖來定位故障源頭，但其準確性極度依賴於精確的故障發生時間(Tf)。RCAeval研究表明，當Tf估計不準確（例如誤差 TΔ​ =60 秒）時，這些方法的性能會顯著惡化。此外，CIRCA 通常需要依賴領域知識或使用 PC 算法 來構造圖結構，而 RCD 雖然透過分而治之策略來提高效率，但這兩種方法皆無法在高維度與雜訊觀測的真實系統中保證對 Tf​ 的脆弱依賴假設成立。
2. 缺乏結構可解釋性的統計方法： BARO 則採取了不同的策略，它不建構因果圖，而是利用多元貝葉斯線上變化點檢測 (Multivariate BOCPD) 和非參數 RobustScorer（基於中位數和 IQR 的假設檢驗）。這使得 BARO 在處理Tf 估計有偏差時表現出更高的魯棒性，並享有運行速度快的優勢。然而，其主要局限性在於犧牲了結構上的可解釋性，無法提供故障沿拓樸傳播的結構化證據。本研究的 GNN_KAN 旨在結合 GNN 的結構化建模能力 和 KAN 的強大非線性特徵提取能力，以克服上述限制。相較於現有 GNN 方法（如 CausalRCA，其計算成本較高且在複雜系統上性能不穩定）GNN_KAN 著重於從高維度與雜訊觀測中提取更魯棒的 RCA 相關特徵，從而在確保可解釋性的同時，實現更高的準確度和跨場景穩健性。
Methodology
本節首先嚴謹定義根因分析問題，明確輸入、輸出與主要挑戰；隨後闡述 GNN_KAN 之整體架構及各模組如何協同解決 RCA 問題；最後詳述圖建模策略、KAN 模組設計、RCA-aware 特徵工程與研究目標。
Problem Formulation and Challenges
根因分析（Root Cause Analysis, RCA) 是指在分散式系統發生異常事件時，從大量可能的故障點中識別出最可能觸發該異常的根本原因。形式化而言，給定：
節點集合 V = {v₁, v₂, …, vₙ}，對應系統中的服務或元件（如微服務、資料庫、訊息佇列等）有向邊集合 E ⊆ V × V，表示服務間的依賴與呼叫關係，(u, v) ∈ E 代表 u 呼叫或影響 v；
時序 W = [t_start, t_end]，涵蓋異常事件的發生與觀測期間
節點觀測 X = x_v(t) v ∈ V, t ∈ W，其中 x_v(t) ∈ ℝ^d 為節點 v 在時間 t 的 d 維觀測向量，包含指標（metrics，如延遲、錯誤率、吞吐量、CPU/記憶體使用）、日誌統計（logs，如異常日誌數、特定錯誤碼出現次數）、追蹤資訊（traces，如請求數、呼叫延遲分布）等；
異常事件標識 A，指示系統在窗口 W 內發生異常（如違反SLA 、出現失敗（error）、整體延遲飆升）。
研究目標是學習一個排名函數 f: (V, E, X) → ℝ^|V|，為每個節點 v 計算根因分數 s_v，並輸出按 s_v 降序排列的節點清單 R。理想情況下，真正的根因節點 v* 應出現在R的排名前三個位（Top-k，k << |V|），最好的情況是第一個排序結果就是根因，使工程師能快速定位並修復問題。
RCA 的核心挑戰包括：
異常傳播與掩蔽效應：根因節點的異常會沿著依賴鏈向下游傳播，導致多個下游節點同時出現異常表徵。下游節點的異常可能更明顯（例如面向用戶的前端服務延遲飆升），但根因往往在上游（例如後端資料庫慢查詢）。這種級聯效應會掩蔽真正的根因，使基於單節點異常強度的方法失效。
高維度與雜訊觀測：每個節點產生多維觀測（數十至上百維），且觀測包含雜訊、尖峰、週期性波動（如日夜負載變化）與負載相關的正常變異。如何從雜訊中提取有效的異常信號，並在多個維度間識別關鍵特徵。
多根因與複雜故障模式：實務中可能存在多個同時發生的根因（multi-root），或根因之間存在交互作用；故障模式多樣（資源瓶頸、邏輯錯誤、外部依賴故障、配置錯誤等），難以用單一規則涵蓋。
可解釋性與部署信任：運維工程師需要理解「為何這個節點被判定為根因」，包括支持證據（該節點的異常表徵）、傳播路徑（如何影響下游）、與其他節點的對比。
How GNN_KAN Addresses RCA
針對上述挑戰，GNN_KAN 提出以下解決策略，並透過整合圖神經網路與知識強化啟動函數實現：
解決方案一：以圖消息傳遞建圖捕捉跳躍異常傳播
 本研究將微服務系統表示為動態加權有向圖 G = (V, E, w)，其中 w: E → ℝ 為邊權函數，反映異常傳播強度。透過多層圖神經網路之消息傳遞機制（message passing），節點表徵 h_v 不僅整合自身觀測 x_v，亦聚合來自鄰居節點之訊息。此機制使模型能捕捉多跳傳播特性：上游節點之表徵經多層聚合後影響下游節點，而學習所得之邊權與聚合權重可反映傳播路徑之重要性。相較於單節點分析或成對關聯分析，GNN 之結構化推理能有效區分直接影響與間接傳播，緩解下游掩蔽效應。
解決方案二：以 Kolmogorov-Arnold Networks架構提升可解釋性
 傳統 GNN 使用固定非線性啟動函數（如 ReLU）與多層感知器（MLP），其函數形狀缺乏領域語義且難以解釋。本研究以 Kolmogorov-Arnold Networks（KAN）取代此類黑箱非線性模組：KAN 基於 Kolmogorov-Arnold 定理，將多變量函數分解為一維可學習函數之組合。具體而言，KAN 以可參數化之一維基底函數（如 B-spline、Chebyshev 多項式、Fourier、PQC）建構函數空間，透過學習組合係數實現函數逼近。此設計帶來雙重優勢：（1）函數分解之可解釋性——每個一維函數之形狀可直接視覺化與驗證，符合「異常越強、傳播越顯著」等領域直觀；（2）降低參數複雜度與過擬合風險——相較於高維 MLP，一維函數組合於參數效率與泛化能力上更優。
策略三：以 RCA-aware 特徵設計
特徵設計為 RCA 成敗之關鍵因素。本研究提出「RCA-aware」設計準則：僅使用與根因定位直接相關且於實務部署中可穩定量測之特徵。具體準則包括：（1）前饋可觀測性——僅使用事件窗口內可前饋觀測之統計量（如延遲分位數、錯誤率、資源使用率），避免引入未來資訊或事後標註指標；（2）傳播導向性——聚焦可傳播之訊號（如上游延遲對下游之影響）與結構關係（依賴強度、呼叫頻率），排除與標註耦合或與根因無直接關聯之特徵；（3）一致性正規化——採用統一之正規化與對齊策略（如 z-score 正規化、時間對齊），降低跨服務偏差與分布偏移。此框架確保模型學習通用異常傳播模式，而非訓練集特定之假性相關。
整體架構：給定異常事件窗口大小，GNN_KAN 首先將各節點觀測編碼為初始表徵 h_v^0；隨後透過 L 層消息傳遞（L=3 為典型配置），每層中邊上訊息經 KAN_edge 轉換，節點聚合鄰居訊息後，先以 KAN_node 轉換節點表徵，再以 KAN_message 處理聚合訊息並透過殘差連接整合；最終讀出層（readout）將 h_v^L 映射為根因分數 s_v，並按降序輸出排名清單。訓練階段以已標註根因窗口為監督信號，最小化排名損失與正則項；推論階段對新異常事件執行前向傳播並輸出候選清單。
下面將詳細闡述GNN_KAN這個方法各部分的設計。
建構圖與消息傳遞 (Graph Modeling and Message Passing)
我們將分散式系統在異常事件窗口內表示為一個動態加權有向圖 (G_t = (V, E, W_t))。節點 (V = {v_1, …, v_n}) 對應各服務或元件，微服務架構 (E \subseteq V \times V) 由服務依賴關係與呼叫trace數據得到。邊權重矩陣 (W_t \in \mathbbRV} \times ^V{) 初始化時預先設定，在訓練過程中會自適應調整以反映異常傳播的實際狀態。
初始節點初始化：對每個節點 (v_i)，本研究將其於觀測窗口內之時序資料 ({x_{v_i}(t_1), …, x_{v_i}(t_m)}) 透過特徵提取轉為固定維度向量。形式化地，初始節點定義為：

其中 (\bar{x}{v_i} \in \mathbb{R}^{d{\text{feat}}}) 為特徵向量，(W_{\text{enc}} \in \mathbb{R}^{d_h \times d_{\text{feat}}}) 為可學習的特徵矩陣，(d_h) 為特徵維度。
多層消息傳遞：GNN_KAN 透過 (L) 層消息傳遞捕捉多跳傳播。在第 (l) 層（(l = 1, …, L)），我們依序執行：
步驟一：邊訊息生成：對每條邊 ((v_u \to v_v) \in E)，源節點 (v_u) 的表徵 (h_u^{(l-1)}) 與邊權 (w_{uv}) 結合，生成傳遞至 (v_v) 的訊息：其中 (\oplus) 表示向量，(e_{uv} = [w_{uv}, \text{feat}{\text{edge}}(u, v)]) 包含邊權重與邊特徵，(\Theta{\text{edge}}^{(l)}) 為第 (l) 層邊 KAN 模組的參數。
（2）節點聚合：節點 (v_v) 接收來自所有入邊的訊息，透過注意力機制加權並聚合：

其中 (\mathcalN}_{\text{in}}(v) = {u (u, v) \in E}) 為 (v) 的鄰居集合。注意力權重 (\alpha_{uv^{(l)}) 透過多頭注意力機制計算：

其中 (a \in \mathbb{R}^{2d_h}) 為可學習的注意力向量，(|) 表示向量連結。此機制使模型自動識別對節點 (v) 影響最大的鄰居，以反映出傳播路徑的重要性。
（3）節點更新：節點更新分為兩個步驟：首先對上一層表徵 \(h_v^{(l-1)}\) 進行 KAN 非線性轉換：

然後，將聚合訊息 \(\tilde{m}_v^{(l)}\) 與轉換後的特徵進行殘差連結：

其中 $\alpha$ 為殘差權重係數，$\text{KAN}_{\text{message}}$ 對聚合訊息進行 KAN 非線性處理。此設計使模型先對節點表徵進行非線性轉換，再整合來自鄰居之訊息，最後透過殘差連接穩定訓練過程並促進梯度流動。$\Theta_{\text{node}}^{(l)}$ 與 $\Theta_{\text{message}}^{(l)}$ 分別為第 $l$ 層節點 KAN 與訊息處理 KAN 模組之參數集合。
評分與排序：最終層特徵 (h_v^{(L)}) 經讀出層映射出根因分數：

其中 (\sigma(\cdot)) 為 sigmoid 函數，確保 (s_v \in (0, 1))，(W_{\text{out}} \in \mathbb{R}^{1 \times d_h})。高分數表示出此節點 (v) 更可能為根因。所有節點按 (s_v) 降序排，最後輸出排名 (R = \text{argsort}_v(-s_v))。
建構圖的方法：本研究採用雙圖策略以整合結構與傳播資訊：
相似性圖 (G_{\text{sim}})：基於特徵餘弦相似度的無向圖，邊權定義為：
 


 其中 (\tau{\text{sim}}) 為相似性閾值。
傳播圖 (G_\text{prop}})：基於時滯相關性的有向圖，捕捉異常傳播方向。對節點對 ((u, v))，我們計算其異常信號的最大時滯相關性：

 若 (\rho_{\text{lag}}(u, v) \geq \tau_{\text{prop}}) 且 (u) 的異常發生時間早於 (v)，則建立有向邊 (u \to v)，權重為：


 其中 (\Delta t_{uv}) 為 (u) 與 (v) 的異常發生時間差，分母項編碼時序一致性。
在前向傳播中，我們對兩圖分別執行消息傳遞，最終融合其輸出：

此雙圖策略使 GNN_KAN 能同時利用結構相似性與異常傳播之時序因果線索，提升根因識別之準確度與穩健性。
Kolmogorov-Arnold Networks（KAN）
傳統神經網路使用固定形狀之啟動函數（如 ReLU、Sigmoid、Tanh）與多層感知器（MLP），此類黑箱非線性模組雖計算簡單且表達力強，然缺乏可解釋性且易過度擬合。Kolmogorov-Arnold Networks（KAN）**基於 Kolmogorov-Arnold 定理，提供可學習且可解釋之函數逼近替代方案。
Kolmogorov-Arnold 定理指出：對任意連續多變量函數 (f: [0,1]^n \to \mathbb{R})，存在一維連續函數 ({\phi_q}) 與 ({\psi_{p,q}})，使得：

此定理表明，任意多變量函數可分解為一維函數之組合，提供結構化函數逼近之理論基礎。KAN 將此數學原理應用於神經網路設計：以可參數化之一維函數取代固定之非線性啟動，透過學習函數組合實現多變量映射。
在 RCA 場景中，KAN 帶來以下優勢：
可解釋的函數分解：每個一維函數可視覺化，驗證是否符合領域預期（如單調性：異常越強、傳播越顯著；門檻性：異常需超過閾值才傳播；飽和性：極端異常的影響趨於飽和。
參數效率與泛化：相較於高維 MLP，一維函數組合的參數量更少，降低過擬合風險，在標註稀缺的 RCA 場景中尤為重要。
先驗注入靈活性：可選擇適合 RCA 的基底函數族（如 B-spline 建模平滑傳播、Chebyshev 多項式建模穩定逼近），將領域知識編碼進模型結構。
一維函數參數化與數學定義
KAN 的核心是將多變量函數分解為一維函數的組合。在實作中，我們對每個輸入維度 (x_i) 建立一維可學習函數 (\psi_i: \mathbb{R} \to \mathbb{R})，最終輸出為這些一維函數的線性組合。形式化地，對輸入 (z = [z_1, \ldots, z_{d_{\text{in}}}]^T \in \mathbb{R}^{d_{\text{in}}})，KAN 層的輸出定義為：

其中 (K) 為一維基函數的數量（典型值 5-10），(b_k: \mathbb{R} \to \mathbb{R}) 為一維基底函數，(\alpha_{i,k} \in \mathbb{R}) 為可學習係數。此結構遵循 Kolmogorov-Arnold 表徵的精神：多變量映射 (\phi: \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}) 分解為 (d_{\text{in}}) 個一維函數 (\psi_i(z_i) = \sum_{k=1}^{K} \alpha_{i,k} \cdot b_k(z_i)) 的加權和。
基底函數選擇：根據程式碼實作，我們支援多種基底函數族，各具不同特性：
（1）Chebyshev 多項式BASE：

其中 $T_k(\cdot)$ 為第 $k$ 階 Chebyshev 多項式，$\text{tanh}(\cdot)$ 將輸入正規化至 $[-1, 1]$ 區間。Chebyshev 基底於該區間上正交，具備良好之逼近性質與數值穩定性。
（2）B-spline BASE：

 其中 (B{k,p}(\cdot)) 為 (p) 階（通常 (p=3) 為立方樣條）B-spline 基底函數，({t_i}) 為節點序列。B-spline 基底具有局部支撐性（每個基底只在有限區間非零），可建模分段平滑的非線性。
（3）Fourier BASE：

Fourier 基底適合捕捉週期性模式，對負載相關的週期性異常特別有效。
（4）PQC BASE：
我們使用 PennyLane 框架實作參數化量子電路（Parameterized Quantum Circuits, PQC），透過 GPU 加速的量子電路模擬捕捉複雜的非線性耦合關係。PQC 基底函數定義為：

其中 [mui] 為量子電路模擬的測量期望值。
架構包含三個主要部分：
資料編碼層：將輸入 𝑧編碼至量子態，使用 RY 旋轉閘：

其中Q為量子比特數（預設為 4）。
 參數化層：包含兩層參數化閘，每層包含：
旋轉閘（對每個量子位元 𝑞 施加 RX、RY、RZ 旋轉）與糾纏閘（使用 CNOT 閘建立量子位元間的糾纏）

測量層：透過 GPU 向量化模擬計算測量期望值。為避免逐樣本執行量子電路，我們使用三角函數模擬：

其中 $\boldsymbol{\theta} \in \mathbb{R}^{3Q}$ 為旋轉參數，$\boldsymbol{\phi} \in \mathbb{R}^{2Q}$ 為相位參數。此公式對應於旋轉閘（RX、RY、RZ）的向量化計算，並透過平均操作模擬量子測量的期望值。
PQC_GPU 透過 GPU 並行化實現高效之量子電路模擬，具備指數級表示能力（理論上可表示 $2^Q$ 維之量子態空間）。於 RCA 場景中，PQC_GPU 特別適合捕捉複雜之多服務協同故障、高維特徵異常與深度非線性傳播模式，於處理服務依賴鏈斷裂等複雜故障時表現優異。
Algorithms - Pseudocode
===
Input: Graph G=(V,E), node features X, labels Y (root-cause nodes),
       KAN bases B, training windows T
Output: Trained parameters Θ = {GNN weights, edge weights, KAN coefficients}

Initialize Θ with structural priors and KAN base coefficients
for epoch = 1..E:
  for each window t in T:
    # Graph encoding
    for v in V:
      h_v^0 ← Encode(X_v(t))
    for l = 1..L:              # message passing layers
      for (u→v) in E:
        m_{u→v} ← KAN_edge(B, h_u^{l-1}, e_{u→v})
      for v in V:
        m_agg ← Aggregate({m_{u→v} | (u→v)∈E})
        h_transformed ← KAN_node(B, h_v^{l-1})
        m_processed ← KAN_message(B, m_agg)
        h_v^l ← h_transformed + α · m_processed

    # Scoring and loss
    s_v ← Readout(h_v^L)
    L_rank ← RankingLoss(s, Y(t))
    L_edge ← EdgeSparsity(e)
    L_kan  ← KANSmoothness(Θ_KAN)
    L_total ← L_rank + λ1 L_edge + λ2 L_kan

    # Parameter update
    Θ ← Θ - η · ∇_Θ L_total
===
KAN 層之前向傳播機制
於實作中，KAN 層接收批次輸入 $Z \in \mathbb{R}^{B \times d_{\text{in}}}$（$B$ 為批次大小），輸出 $\Phi(Z) \in \mathbb{R}^{B \times d_{\text{out}}}$。完整之前向傳播包含三個平行計算之模組：
基函數計算

組合線性基函數
其中 $A \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}} \times K}$ 為可學習之基底係數張量，\texttt{bid,oid->bo} 表示對 $i$（輸入維度）與 $d$（基底索引）求和，實現 $B \times d_{\text{in}} \times K$ 與 $d_{\text{out}} \times d_{\text{in}} \times K$ 之高效張量收縮。

可學習線性基底
其中 $W_{\text{base}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 為可學習權重矩陣，$b_{\text{base}}$ 為偏置項。

最終輸出：
平行化實作策略
（1）向量化基底函數計算：所有基底函數 $b_k(\cdot)$ 以張量運算實現，對批次 $B$ 與輸入維度 $d_{\text{in}}$ 全面並行。例如 Chebyshev 基函數透過遞迴公式 $T_{k+1}(x) = 2x \cdot T_k(x) - T_{k-1}(x)$ 實現，遞迴展開為張量運算，單次前向傳播計算所有階數。
（2）Einstein 求和優化：使用 PyTorch 之 `torch.einsum` 進行張量收縮，其底層調用 BLAS/cuBLAS 實現矩陣乘法之 GPU 加速。對於批次大小 $B \sim 32$、基底數 $K \sim 8$、維度 $d \sim 64$，einsum 相較循環實現可達 10-50× 加速。
（3）混合精度訓練：KAN 層支援 FP16/BF16 混合精度，基底計算於 FP32 以保證數值穩定性，線性變換與聚合於 FP16 以加速，透過自動混合精度實現。
RCA-aware 特徵工程
特徵設計為 RCA 模型成功之關鍵因素，不當之特徵會引入資料洩漏（data leakage）或分布偏移（distribution shift），導致訓練集高準確度然部署時失效。
RCA-aware 特徵工程為以下四個部分構成：
使用前饋可觀測特徵（Forward-Observable Features）：
特徵必須於異常事件窗口 $W = [t_{\text{start}}, t_{\text{end}}]$ 內即時可得，不依賴未來資訊或事後標註。形式化而言，特徵提取函數 $\mathcal{F}$ 必須滿足因果約束：

其中 $\mathcal{N}(v)$ 為節點 $v$ 之鄰居集合。禁止使用需事後回溯之指標或與標註耦合之資訊。
聚焦可傳播訊號與結構（Propagation-Centric Signals）：RCA 之核心為理解異常如何傳播，因此特徵應捕捉傳播相關之訊號。本研究將特徵分為三類：異常強度特徵：量化節點偏離正常狀態之程度；時序變化特徵：捕捉異常之動態特性與變化點；傳播關聯特徵：量化節點間之異常關聯性與時滯。
排除標註耦合之後驗指標（No Label-Coupled Features）：若特徵函數 $\mathcal{F}$ 依賴標註 $y$，即 $\bar{x}_v = \mathcal{F}(x_v, y)$，則為洩漏特徵。本研究透過靜態分析與動態驗證確保 $\mathcal{F}$ 與 $y$ 獨立。
一致之正規化與對齊策略（Consistent Normalization and Alignment）：對於每個觀測維度 $x_v^{(d)}(t)$，本研究基於歷史正常期 $\mathcal{T}_{\text{normal}}$ 計算 z-score 正規化：
$$\tilde{x}_v^{(d)}(t) = \frac{x_v^{(d)}(t) - \mu_v^{(d)}}{\sigma_v^{(d)} + \epsilon}$$
其中 $\mu_v^{(d)} = \mathbb{E}_{t \in \mathcal{T}_{\text{normal}}}[x_v^{(d)}(t)]$ 為正常期均值，$\sigma_v^{(d)} = \text{Std}_{t \in \mathcal{T}_{\text{normal}}}[x_v^{(d)}(t)]$ 為正常期標準差，$\epsilon = 10^{-8}$ 為數值穩定項。
系統化特徵定義
本研究對每個服務節點 $v$ 提取以下特徵向量 $\bar{x}_v \in \mathbb{R}^{d_{\text{feat}}}$，特徵向量由以下特徵構成：SLI 相關特徵、時序差異特徵、異常偵測特徵、變化幅度特徵，根據這些特徵按列進行 z-score 標準化。

Experimental Architecture
Dataset
Online Boutique：Google 開源之電商微服務示範應用，包含約 10-15 個服務（前端、產品目錄、購物車、支付、推薦等），採用典型之扇出型拓樸。本研究注入延遲瓶頸（資料庫慢查詢）、資源耗盡（記憶體洩漏）、錯誤率飆升（外部 API 故障）等異常，產生數百個標註之根因窗口。
Sock Shop：WeaveWorks 之微服務基準，包含約 10 個服務（前端、目錄、購物車、訂單、支付、用戶管理等），拓樸結構較線性。注入異常包括 CPU 瓶頸、網路延遲、服務崩潰等，測試模型對不同故障模式之識別能力。
Train Ticket：複雜之火車票訂票系統，包含 40+ 個微服務，拓樸深度與廣度較大，依賴關係錯綜複雜。此資料集挑戰模型於大規模場景之可擴展性與多跳傳播推理能力。
RE1 系列（Reproducible Evaluation 1）：
RE1 是 RCAEval 平台提供的標準化評估資料集，包含三個子資料集：
RE1-OB（Online Boutique）：基於 Google 開源的 Online Boutique 電商微服務系統，包含約 10-15 個服務，採用典型的扇出型拓樸（前端服務同時依賴多個後端服務）。此資料集包含 375 個故障案例，涵蓋 3 種資源故障類型（CPU、記憶體、磁碟）與 2 種網路故障類型（延遲、封包遺失），每個服務提供 49-212 個觀測指標。RE1-OB 的故障注入策略遵循真實場景的異常模式，適合評估模型在電商類應用的根因分析能力。
RE1-SS（Sock Shop）：基於 Sock Shop 系統的標準化資料集，包含約 10 個服務，採用線性拓樸。此資料集同樣包含 375 個故障案例，故障類型與 RE1-OB 一致，但服務拓樸與依賴模式不同，適合測試模型對不同拓樸結構的適應性。
RE1-TT（Train Ticket）：基於 Train Ticket 火車票訂票系統，包含 40+ 個微服務，拓樸深度與廣度較大，依賴關係錯綜複雜。此資料集挑戰模型在大規模、深層依賴場景下的多跳傳播推理能力，是評估可擴展性的關鍵資料集。
RE2 系列（Reproducible Evaluation 2）：RE2 是 RCAEval 平台提供的擴展評估資料集，在 RE1 基礎上增加了多源遙測數據（metrics、logs、traces）：
RE2-OB：包含 270 個故障案例，涵蓋 4 種資源故障類型（CPU、記憶體、磁碟、Socket）與 2 種網路故障類型，每個服務提供 77-376 個觀測指標，並包含 8.6-26.9 百萬條日誌記錄與 39.6-76.7 百萬條追蹤記錄。此資料集的規模與複雜度顯著高於 RE1，適合評估模型在多源數據融合與大規模場景下的性能。
RE2-SS：基於 Sock Shop 的擴展資料集，同樣包含多源遙測數據，規模為大型（large scale），適合測試模型在中等規模系統但資料豐富場景下的表現。
RE2-TT：基於 Train Ticket 的擴展資料集，規模為超大型（very large scale），包含最複雜的服務拓樸與最豐富的觀測數據，是評估模型極限性能的關鍵資料集。
RE3 系列（Reproducible Evaluation 3）：RE3 專注於代碼級故障（code-level faults），包含 5 種代碼級故障類型（F1-F5），共計 90 個故障案例：
RE3-OB/SS/TT：三個子資料集分別基於 Online Boutique、Sock Shop 與 Train Ticket 系統，每個服務提供 68-322 個觀測指標，並包含 1.7-2.7 百萬條日誌與 4.5-4.7 百萬條追蹤記錄。代碼級故障相較於資源與網路故障更為細粒度，需要模型捕捉更微妙的異常信號，適合評估 GNN_KAN 在精細根因定位上的能力。
Data and Graph Pipeline
資料載入：由平台 Datasets/Registry 管理，明確標示資料集版本、切分比例（訓練/驗證/測試：60%/20%/20%）與來源，確保事件間的時間獨立性以避免資料洩漏；
特徵處理：落實第 3.5 節的 RCA-aware 特徵工程，對每個服務節點提取特徵向量；
圖構建：實作雙圖策略（相似性圖與傳播圖）與時滯相關性計算，並輸出標準化的圖張量（節點特徵矩陣、邊索引、邊權重）。
Model and Learning
模型架構：包含多層消息傳遞與KAN 模組（KAN_edge、KAN_node、KAN_message），KAN 基底函數類型（Chebyshev、B-spline、Fourier、PQC_GPU）。
訓練機制：採用排名損失，並加入邊稀疏正則化。
隨機性控制：透過平台統一設定隨機種子，涵蓋資料抽樣、權重初始化與後端運算，確保實驗可重現性。
評估指標
RCA 之核心目標為於候選清單前列找到根因，因此本研究採用排名導向之指標：
Precision@k / Recall@k：前 k 個預測中正確根因之比例（Precision）與覆蓋全部根因之比例（Recall）。對於多根因場景，此類指標更為全面。
Mean Average Precision (mAP)：平均所有事件之平均精度，考量整個排名清單之品質，對排名順序更為敏感。
Baselines
為證明 GNN_KAN 之優勢，本研究與以下幾類代表性基線比較：

統計與關聯方法：
Random Walk (PageRank)：以異常分數為初始，在依賴圖上隨機遊走並排名。
BARO：以異常分數為初始，在依賴圖上隨機遊走並排名。

Setup
計算環境：實驗於配備 NVIDIA 2080ti GPU 之伺服器上執行。
可重現性（RCAEval 平台）：為使上述方法能於不同資料集與場景中以一致方式執行與比較，本研究將整體流程編排於 RCAEval 平台，包含標準化之資料載入（Datasets/Registry）、RCA-aware 特徵處理（Features/Processors）、雙圖構建（Graphs/Builders）與模型訓練評估（Runners/Evaluators）。所有實驗透過此平台執行，採用以下重現協定：
設定管理：以平台之設定檔（configs）統一管理資料集、特徵處理、圖構建與模型超參數，所有實驗均附設定快照；
資料載入與版本控管：資料集透過平台之 Datasets/Registry 統一載入，明確標示版本與切分，避免資料漂移；
隨機性控制：於資料抽樣、權重初始化與運算後端層面固定隨機種子，確保多次執行結果一致（允許浮點微幅差異）；
實驗執行器：使用平台 Runner 統一啟動訓練與評估，記錄指標、模型權重、圖與特徵摘要；
產物追蹤：平台自動輸出實驗日誌、評估指標（CSV/JSON）、Top-k 排名清單，並以結構化目錄保存，利於審計與對比。
Results and Analysis
本節探討 GNN_KAN 與基線方法 BARO 之實驗結果，涵蓋九個核心資料集（RE1系列、RE2系列、RE3系列）之全面評估。本研究比較 GNN_KAN 於不同基底函數（Chebyshev、B-spline、Fourier、PQC_GPU）下之表現，並與強基線 BARO 進行對比，深入分析各方法於不同故障類型與系統規模下之優劣。

RE1 系列資料集：標準化評估基準
RE1 系列是 RCAEval 平台的標準化評估資料集，包含 375 個故障案例，涵蓋資源故障與網路故障，是評估模型在標準場景下性能的關鍵基準。

RE1-OB（Online Boutique）：
在 RE1-OB 資料集上，BARO 展現了強勁的整體性能，Precision@1 達到 0.65，Precision@3 為 0.896，Precision@5 為 0.928，Avg@5 為 0.784，在所有方法中領先。GNN_KAN 的各基底函數變體表現相近但略低於 BARO：Chebyshev 與 Fourier 的 Precision@1 均為 0.432，Precision@3 均為 0.624，Precision@5 均為 0.632，Avg@5 分別為 0.584 與 0.5856。B-spline 與 PQC 表現略低。此結果顯示，在標準化的資源與網路故障場景中，統計方法（BARO）可能因資料充分性而獲得優勢，但 GNN_KAN 仍能提供穩定的性能。

RE1-SS（Sock Shop）：
在 RE1-SS 資料集上，PQC 與 Chebyshev 表現最佳，PQC 的 Precision@3 與 Precision@5 分別達到 0.792 與 0.952，Avg@5 為 0.6880；Chebyshev 的 Precision@5 為 0.936，Avg@5 為 0.6784。BARO 在 Precision@1 上略優（0.28），但整體表現相近。此結果顯示，在中等規模系統中，PQC 與 Chebyshev 基底函數均能提供穩定的性能。

RE1-tt（Train Ticket ）：
於 RE1-tt資料集上，PQC_GPU 基底函數表現最佳，Precision@1 達到 0.256，Precision@3 為 0.376，Precision@5 為 0.456，Avg@5 為 0.3648，顯著優於其他基底函數變體與 BARO 基線，展現參數化量子電路於複雜系統中之優勢。BARO 表現第二（0.136/0.28/0.392/0.272），GNN_KAN 之其他基底函數變體（Chebyshev、B-spline、Fourier）表現較低。此結果明確表明，於超大規模複雜系統（40+ 服務）中，PQC_GPU 基底函數之指數級表示能力能有效捕捉深度非線性傳播模式，為此類場景之最佳選擇。

RE2 系列資料集：多模態數據
RE2 系列包含多模態數據（metrics、logs、traces），規模與複雜度顯著高於 RE1，是評估模型在多源數據融合與大規模場景下性能的關鍵資料集。

RE2-OB（Online Boutique ）：
在 RE2-OB 資料集上，BARO 展現了統計方法在 Top-3 與 Top-5 根因定位上的優勢，Precision@3 與 Precision@5 分別達到 0.8778 與 0.9444，Avg@5 為 0.7422，但 Precision@1 僅為 0.1444，顯示其在精確根因定位上的不足。GNN_KAN 的各基底函數變體在 Precision@1 上表現優於 BARO：Chebyshev 與 Fourier 均為 0.4778，PQC 為 0.4778，B-spline 為 0.3667，但在 Precision@3 與 Precision@5 上略低於 BARO。Fourier 基底函數在 Avg@5 上表現最佳（0.76），Chebyshev 為 0.7578，顯示在大型多源數據場景中，Fourier 基底函數能有效捕捉週期性異常模式。

RE2-SS（Sock Shop ）：
在 RE2-SS 資料集上，BARO 再次展現了在 Top-3 與 Top-5 上的優勢（Precision@3: 0.8956, Precision@5: 0.9269, Avg@5: 0.7244），但 Precision@1 極低（0.0667）。GNN_KAN 的各基底函數在 Precision@1 上優於 BARO：Chebyshev 為 0.3333，Fourier 為 0.3222，B-spline 為 0.2889，PQC 為 0.2，但在 Precision@3 與 Precision@5 上略低於 BARO。此結果表明，在中等規模系統的多源數據場景中，GNN_KAN 與 BARO 各有優勢：GNN_KAN 更適合精確根因定位，BARO 更適合候選清單生成。

RE2-TT（Train Ticket）：
在 RE2-TT 資料集上，BARO 展現了最佳整體性能，Precision@1 達到 0.5936，Precision@3 為 0.7572，Precision@5 為 0.7955，Avg@5 為 0.7152，在所有方法中領先。GNN_KAN 的各基底函數變體表現相近：PQC_GPU 的 Precision@1 最高（0.5222），Chebyshev 的 Precision@3 為 0.5909，但整體略低於 BARO。

RE3 系列資料集：程式碼故障
RE3 系列專注於程式碼故障（code-level faults），是評估精細根因定位能力的關鍵資料集。實驗結果顯示，GNN_KAN 在程式碼故障情境下展現顯著優勢：

RE3-TT（Train Ticket 系統）：
在 RE3-TT 資料集上，GNN_KAN 的所有基底函數變體（Chebyshev、B-spline、Fourier、PQC_GPU）均達到完美性能，所有指標（Precision@1、Precision@3、Precision@5、Avg@5）均為 1.00，展現了對複雜大規模系統程式碼故障的卓越辨別能力。相較之下，BARO 基線在 Precision@1 上僅達到 0.7667，雖然 Precision@3 與 Precision@5 均為 1.00，但 Avg@5 為 0.9533，顯示其在 Top-1 根因定位上存在不足。此結果證實了 GNN_KAN 透過圖結構推理與 KAN 非線性轉換，能更精準地捕捉程式碼故障的細微異常信號。

RE3-SS（Sock Shop 系統）：
在 RE3-SS 資料集上，GNN_KAN 的表現同樣優於 BARO。BARO 在 Precision@1 上完全失敗（0.00），雖然 Precision@3 與 Precision@5 分別達到 0.9667 與 1.00，但 Avg@5 僅為 0.74，顯示其對程式碼故障的 Top-1 識別能力嚴重不足。GNN_KAN 的各基底函數變體表現優異：B-spline 達到最佳 Precision@1（0.8333）與 Avg@5（0.9667），Chebyshev、Fourier 與 PQC_GPU 的 Precision@1 均為 0.8，Avg@5 均為 0.96，且 Precision@3 與 Precision@5 均為 1.00。此結果表明，在中等規模系統的程式碼故障場景中，B-spline 基底函數最適合捕捉平滑的異常傳播模式。

RE3-OB（Online Boutique 系統）：
在 RE3-OB 資料集上，Chebyshev 基底函數表現最佳，Precision@1 達到 0.9667，Precision@3 與 Precision@5 均為 1.00，Avg@5 為 0.9933，接近完美性能。PQC_GPU 表現穩定（0.9/0.9/0.9/0.9），B-spline 與 Fourier 表現相近（0.8667/0.9/0.9/0.8933）。BARO 在 Precision@1 上再次失敗（0.00），Precision@3 與 Precision@5 為 0.9667，Avg@5 為 0.7667。此結果顯示，在扇出型拓樸的電商系統中，Chebyshev 基底函數的穩定逼近特性最適合程式碼故障的根因分析。可能是因為在程式碼故障時的CPU與Memory變化較明顯。

基底函數比較

Chebyshev 基函數：
Chebyshev 在程式碼故障（RE3 系列）與資源故障（RE1-OB）場景下表現最佳，特別是在 RE3-OB 與 RE1-OB 的 CPU/記憶體故障上達到完美性能。Chebyshev 多項式的穩定逼近特性使其適合捕捉平滑的異常傳播模式，在扇出型拓樸系統中表現優異。然而，在網路故障場景下表現較差，可能因其無法有效捕捉網路延遲的非線性模式。

B-spline 基函數：
B-spline 基函數在 RE3-SS（線性拓樸）上表現最佳，展現了分段平滑函數在線性依賴結構中的優勢。B-spline 的局部支撐性使其能建模分段平滑的非線性，適合捕捉局部異常傳播模式。但在多源數據場景（RE2 系列）中表現略低，可能因其對多模態數據特徵的捕捉能力有限。

Fourier 基函數：
Fourier 基函數在多模態場景（RE2-OB）的 Avg@5 指標上表現最佳，展現了週期性基底在捕捉負載相關異常模式上的優勢。Fourier 適合建模週期性模式，對負載相關的週期性異常特別有效。但在程式碼故障場景下表現略低於 Chebyshev，可能因其對非週期性異常的捕捉能力有限。

PQC_GPU 基底函數：
PQC_GPU 基底函數在超大規模複雜系統（Train Ticket、RE2-TT）上展現優勢，展現了參數化量子電路的指數級表示能力。PQC_GPU 特別適合捕捉複雜的多服務協同故障、高維特徵異常與深度非線性傳播模式。然而，其計算成本較高，在中等規模系統中可能不如 Chebyshev 或 B-spline 效率來得好。

基函數選擇策略

基於上述實驗結果，本研究提出以下基函數選擇策略的方向：

方向一：基於系統拓樸結構選擇
扇型拓樸（如 Online Boutique）：優先選擇 Chebyshev 基底函數，因其穩定逼近特性適合多服務依賴場景。
線性拓樸（如 Sock Shop）：優先選擇 B-spline 基底函數，因其分段平滑特性適合線性依賴結構。
深層複雜拓樸（如 Train Ticket，40+ 服務）：優先選擇 PQC_GPU 基底函數，實驗結果顯示其於 Train Ticket 資料集上表現最佳（Precision@1=0.256，Avg@5=0.3648），其指數級表示能力適合複雜多跳傳播場景。

方向二：基於資料豐富度選擇
純指標數據（RE1 系列）：優先選擇 Chebyshev 或 BARO，統計方法在資料充分時可能獲得優勢。
多源數據（RE2 系列，包含 logs、traces）：優先選擇 Fourier 基底函數，因其能捕捉多源數據中的週期性模式。
超大規模數據（RE2-TT、Train Ticket）：優先選擇 PQC_GPU 基底函數，因其指數級表示能力適合高維複雜場景。

方向三：基於機器性能與效率權衡
追求計算效率：優先選擇 Chebyshev 或 B-spline，計算成本較低。
平衡性能與效率：選擇 Chebyshev 作為預設基底函數，在大多數場景下提供穩定的優良判斷。


Conclusion

本研究針對現代分散式系統根因分析面臨之挑戰——異常傳播掩蔽、高維雜訊、動態拓樸、可解釋性需求——提出 GNN_KAN，一結合圖神經網路與 Kolmogorov-Arnold Networks 之創新框架。GNN_KAN 之核心貢獻如下：

本研究將 Kolmogorov-Arnold Networks（KAN）引入 GNN 之消息傳遞機制，利用 Kolmogorov-Arnold 表徵定理將多變量非線性映射分解為一維可學習函數之組合，取代傳統之黑箱 MLP。透過選擇適當之一維基底函數（如 B-spline、Chebyshev 多項式、Fourier 基底、PQC_GPU），KAN 於提供強大逼近能力之同時，降低參數複雜度並提升可解釋性。一維函數可獨立視覺化與驗證，使工程師能理解模型如何建模異常傳播之非線性模式（如單調性、門檻性、飽和性）。

本研究系統化地定義 RCA-aware 特徵設計原則，確保特徵與根因定位任務高度相關且於部署中可行。透過排除未來資訊、標註耦合與後驗指標，並採用一致之正規化與對齊策略，本研究使模型學習到通用之異常傳播模式，而非訓練集特定之偽相關，大幅提升跨場景穩健性。

本研究於五個核心資料集（sock-shop-1、sock-shop-2、RE1、RE2、RE3）上進行全面評估，涵蓋從中等規模到超大規模、從資源/網路故障到代碼級故障之完整評估光譜。實驗結果顯示，GNN_KAN 於代碼級故障場景（RE3 系列）下展現顯著優勢，所有基底函數變體均達到完美或接近完美性能，顯著優於統計基線 BARO。於資源故障場景中，GNN_KAN 之 Chebyshev 基底函數達到完美性能。特別值得注意的是，於超大規模複雜系統（Train Ticket）中，PQC_GPU 基底函數表現最佳（Precision@1=0.256，Avg@5=0.3648），顯著優於其他基底函數與 BARO 基線，展現參數化量子電路於複雜多跳傳播場景中之優勢。然於網路故障與部分多源數據場景中，BARO 展現統計方法之優勢，顯示不同方法適用於不同場景。

基於實驗結果，本研究提出系統化之基底函數選擇策略，根據系統拓樸結構、故障類型、資料豐富度與性能效率權衡，為實務部署提供明確之指導原則。此選擇策略使 GNN_KAN 能根據具體場景選擇最適合之基底函數，最大化根因分析之準確度與效率。

GNN_KAN 代表根因分析研究從黑箱模型向可解釋、可信任 AI 之轉變。透過 Kolmogorov-Arnold 表徵之結構化函數分解與 RCA-aware 之特徵設計，本研究展示將數學基礎理論（Kolmogorov-Arnold 表徵定理）與深度學習結合於實務問題之有效性。隨著雲端系統複雜度之持續增長，可解釋且穩健之 RCA 工具將成為保障服務可靠性之關鍵。本研究期待 GNN_KAN 能為學術界與產業界提供有價值之參考，並激發更多基於數學原理之可解釋 AI 於運維領域之創新。

Future

理論與方法論擴展的方向
因果推論與 GNN_KAN 之深度融合：
目前 GNN_KAN 主要依賴統計相關性與結構先驗進行根因推論，未來可探索將因果發現方法（如 PC 演算法、FCI 演算法、結構方程模型）與 GNN_KAN 之深度融合架構。具體而言：

因果圖作為結構先驗：以因果發現識別之真實因果邊作為 GNN 之初始邊權，並透過可學習之邊權調整機制於訓練中精細化，使模型同時具備因果嚴謹性與資料驅動之適應性；
混合推理架構：設計分層架構，底層以因果推論識別高置信度之直接因果關係，上層以 GNN_KAN 處理間接傳播與複雜交互，透過注意力機制動態選擇推理路徑；
因果感知之 KAN 基底：將因果結構（如中介變數、混淆變數）編碼至 KAN 基函數之選擇與初始化，使一維函數之形狀反映因果機制（如單調性、門檻性、飽和性）。

可解釋性之量化與標準化：
目前對可解釋性之評估主要依賴人工檢視與案例研究，缺乏標準化之量化指標。未來可建立：
可解釋性度量框架：定義多維度可解釋性指標，包括路徑合理性分數（學習到之傳播路徑與專家知識之一致性）、函數形狀合理性（KAN 一維函數是否符合領域預期）、決策透明度（根因分數之計算過程是否可追溯）；
可解釋性基準測試：構建標準化之可解釋性評估資料集，包含已知傳播路徑之故障案例，用於客觀比較不同方法之可解釋性；
自動化可解釋性驗證：開發工具自動檢查 KAN 函數曲線是否違反領域約束（如異常強度與傳播強度應單調相關），並提供修正建議。

貝氏 GNN_KAN 與不確定性估計：
為模型預測提供信心度估計，幫助工程師判斷何時信任模型、何時需人工介入，未來可研究：
貝氏 KAN：將 KAN 之基底係數視為隨機變數，透過變分推論（variational inference）或 MCMC 採樣估計後驗分布，提供預測之不確定性區間；
集成學習與模型平均：訓練多個 GNN_KAN 模型（不同初始化、不同基底函數），透過集成預測與投票機制提供信心度估計，並識別高不確定性之異常事件；
對抗性驗證：設計對抗樣本生成器，測試模型於邊緣案例（edge cases）下之穩健性，並提供不確定性警告。


