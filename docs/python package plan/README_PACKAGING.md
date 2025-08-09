# GNN+KAN 封裝與發佈指南（Packaging Readme）

本資料夾收錄將現有 GNN+KAN 方法封裝為獨立 Python Package 的規劃與規格文件：

- `GNNKAN_Package_Plan.md`：整體藍圖、目錄結構、API、測試與發佈計畫
- `Input_Standard_Spec.md`：輸入通用性規格與驗證規則
- `Refactor_Checklist.md`：重構檢核清單（去重、命名一致、型別、穩定性、測試）
- `Domain_Adapters_Spec.md`：跨領域（分子化學、量子物理）Adapter 介面

## 下一步（建議）
1. 釐清授權策略與套件名稱（預設：`gnnkan`）
2. 依 `Refactor_Checklist.md` 完成去重與 API 一致化（M1）
3. 建立 `gnnkan/` 目錄並逐步遷移（不改邏輯）
4. 撰寫最小單元測試與 `examples/rca_example.py`
5. 設定 CI 並打包 wheel；內部先測後再公開

聯絡方式：直接於 PR/Issue 留言或在專案維護群組內同步。 