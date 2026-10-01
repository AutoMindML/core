# AGENTS Instructions

## 專案邊界與修改位置

- 本文件適用整個 AutoMind Core 儲存庫。修改前先看 `git status --short`，保留既有變更；不要以重設、清理或覆寫輸出取代調查。
- `automind/` 與 `api/` 是各有 `pyproject.toml`、`uv.lock`、`.venv` 的 Python 3.10 專案；根目錄
  `justfile` 只包裝各自的指令。API 以 editable 路徑引用本地 `../automind/`，共用行為應修改核心套件。
- `automind/src/automind/data_utils/` 與 `models/` 負責資料融合、欄位解析、舊版 DC／FE 方法及共用
  schema；`pipeline/` 負責 fitted 前處理、語意驗證與候選選擇；`service/` 負責共用 LLM 設定、provider 與回應契約。
- `automind/src/automind/experiments/`
  負責資料集適配、protocol、執行、評估、sandbox、進度與產物；`configs/llm/` 與 `configs/research/`
  是版本化設定來源。TPOT 封裝位於 `engine/`，評估邏輯亦見 `evaluation/`。
- `api/src/automind_api/app/` 中，`controllers/` 定義路由，`services/` 整合流程，`repositories/`
  封裝存取，`models/` 定義請求與資料庫映射；`db/`、`configs/` 處理連線及設定，`api/create-db/` 擁有 SQL schema、view
  與 procedure。核心套件不得反向依賴 API。
- 路由掛載以 `api/src/automind_api/main.py` 為準；資料庫 view／procedure 名稱以
  `api/src/automind_api/app/models/view_sp.py` 與 SQL 為準。修改 API／MindsDB 啟動時同時檢查
  `api/start.py`、`api/__main__.py`；其 `../../mindsdb/` 是儲存庫外的環境。

## 資料、LLM 與實驗契約

- 新增或修改舊版 DC／FE 方法時，同步檢查 `models/preprocessing.py` 的
  enum／schema、`data_utils/template.py` 的提示詞與範例、`data_utils/preprocessing.py` 的
  `@register_method`、`data_utils/logic_applier.py` 的分派，以及測試與核心 README。EnumByName 與方法
  registry 以 enum **名稱**為鍵，不任意改名或引入衝突。
- 舊版 LLM 建議使用 `<json>...</json>`；改格式時同步檢查 parser、schema、提示詞與 fixture。保留 `LogicApplier`
  的輸入 DataFrame 複製、原始／處理後資料分離，以及索引、目標和特徵對齊。
- fitted pipeline 只在訓練資料擬合，抽樣只作用於訓練分割；驗證／holdout 使用固定狀態與特徵 schema。變更 `pipeline/`
  或實驗比較流程時，測試洩漏防護、欄位對齊及不受支援建議的失敗行為。
- 一般 LLM 建議應經 schema 與已註冊方法處理，不直接 `eval` ／`exec` 回應文字。研究用 direct code 僅能透過受控 sandbox
  執行，不在主機 Python 程序執行；保留 row ID、target、單列 holdout、唯讀 transform state 與網路隔離契約。
- Protocol 檔名採 `<dataset>-<experiment>-v<schema>.protocol.json`；檔名可調整，但 JSON 的
  `name`、`output_root` 等內容參與 fingerprint。修改這些欄位須使用新執行身分與輸出位置，不讓 `resume`
  接受不相符的舊產物。中斷標記、attempt journal 與失敗產物應保留供診斷；不要手改結果來偽裝成功。

## API 與設定契約

- 修改 stored procedure 時同步檢查 Python parameter mapping；`exec_mutation_sp()` 的參數名稱須與 SQL
  一致。修改 view 時同步檢查 TypedDict 欄位，`get_view_by_id()` 依 annotations 建立查詢。
- 保留 HTTP 回應中 `state`、`message`、`new_id` 的語意，並檢查 controller、model 與呼叫端。SQL
  值使用綁定參數；動態識別名稱必須來自受控映射或驗證。
- `HeaderSessionMiddleware` 由 `X-User-Id` 建立 request state；`env == "test"`
  使用固定身分。不要將正式設定改成 test 來繞過驗證；部署 API 的金鑰驗證也須保留。
- API 設定依執行工作目錄讀取 `src/automind_api/configs/`；部分舊 Azure 模組會在匯入時建立 client。共用 LLM 設定由
  `service/config.py` 解析 profile、環境變數及 `AUTOMIND_ENV_FILE`。核心 `utils/config.py`
  也依目前工作目錄定位設定；改路徑前先檢查呼叫端。
- `.env`、`private.*`、server／MindsDB 設定與 SQL／batch
  腳本可能含機密或機器資訊。只讀必要結構，不將值印入工具輸出、文件、測試或提交；若發現意外追蹤的機密，同樣保護。

## 驗證與副作用

- 執行 `just` recipe 前查看 `justfile`
  的工作目錄及副作用。`just sync`、`just test`、`just lint`、`just typecheck`
  可能同步兩個環境；文件修改不因此執行安裝、服務啟動或模型訓練。
- 核心隔離測試在 `automind/`
  執行：`uv run --locked --group dev --no-sync -m pytest ./src/automind/tests -m "not live_llm and not tpot_smoke and not podman_sandbox" -x`。既有環境與
  `src/automind/data/csv/synthea_covid19_10k/` 的本機 CSV fixture
  必須存在；資料未納入版本控制，不以空檔替代。部分測試會寫入 `llm_query.txt`。
- API 隔離測試由根目錄 `just api-test` 選取；`api/src/automind_api/tests/`
  同時有手動整合腳本，不整目錄當作離線單元測試。新增 API 測試時隔離資料庫、MindsDB、LLM、lifespan 與匯入時初始化。
- 修改跨模組資料流程或 schema 後執行相關聚焦測試與核心隔離測試；sandbox 契約變更另執行 `just experiment-sandbox-check`。文件與
  protocol 整理可用明確路徑的 validate、連結檢查及 `git diff --check`；v2 dry-run 仍執行 Podman
  preflight，不視為純靜態檢查。缺少前提時說明未執行的檢查，不把靜態檢查說成測試通過。
- `just api-live` 會連接 SQL Server 並啟動外部 MindsDB；`just experiment-run-live`／`just experiment-resume-live` 會發出真實
  LLM 請求、訓練並寫入產物。未有相應授權，不執行資料庫初始化／DDL、遠端刪改、引擎上傳或真實 Azure 請求；沿用當前對話已有的授權。
- `api/create-db/` 的 batch／SQL 可能修改資料庫與帳號，部分 SQL 啟用 `TRUSTWORTHY`；`api/db-setup.bat`
  仍引用不存在的 sqlpredictor 路徑。實驗可能覆寫 CSV、報表及模型；測試使用 mock 與暫存目錄，不載入不受信任的 pickle。
- 不手改 `.venv/`、`*.egg-info/`、cache、產生的 typings 或二進位 DLL，不順帶重產研究輸出。對外流程變更更新對應
  README；sandbox 操作歸
  `automind/docs/direct-code-sandbox.md`，代理編輯規則留在本文件。提交前確認只包含任務檔案，沒有機密、資料或模型產物。
