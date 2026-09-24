# AGENTS Instructions

## 適用範圍與專案邊界

- 本文件適用於整個 AutoMind Core 儲存庫。這是獨立 Git 儲存庫；在此使用一般 `git` 指令，上層家目錄的 dotfiles 專用 bare Git 指令不適用於本專案。
- 修改前檢查 `git status --short`，保留使用者既有變更；未經明確授權，不使用破壞性的重設或清理指令。
- 這裡有兩個 Python 套件，各自擁有 `pyproject.toml`、`uv.lock`，並使用各自的 `.venv`；根目錄沒有 Python 專案設定。兩個 `.python-version` 均指定 Python 3.10。
- `api/pyproject.toml` 透過 `[tool.uv.sources]` 以 editable 模式引用 `../automind/`。修改共用行為時直接修改本地核心套件，不另裝同名發布套件來取代它。

## 程式碼歸屬

| 路徑 | 修改責任 |
| --- | --- |
| `automind/src/automind/data_utils/` | 資料融合、欄位解析、metadata／提示詞產生、LLM 建議解析與前處理執行 |
| `automind/src/automind/models/` | 前處理 schema、enum、方法註冊與共用資料契約 |
| `automind/src/automind/engine/`、`automind/src/automind/evaluation/` | TPOT 訓練／預測封裝與實驗評估 |
| `api/src/automind_api/app/` | `controllers/` 定義 HTTP 路由，`services/` 整合流程，`repositories/` 封裝資料存取，`models/` 定義請求與資料庫映射 |
| `api/src/automind_api/db/`、`api/src/automind_api/configs/` | SQL Server／MindsDB 連線與 API 設定 |
| `api/create-db/` | I3S 與 AutoMind 的 SQL schema、view、stored procedure 與初始化腳本 |

- 可重用的 DataFrame、特徵工程與前處理邏輯放在 `automind`；HTTP、session 與資料庫整合放在 `automind_api`。不要讓核心套件反向依賴 API。
- 路由掛載以 `api/src/automind_api/main.py` 為準；資料庫 view／procedure 名稱以 `api/src/automind_api/app/models/view_sp.py` 與對應 SQL 為準，不在文件另建完整清單。
- `automind/src/automind/experiments/` 是目前版本化的實驗、報表與資料準備入口；已移除的 `_experiment/` 與 `_analysis/` 路徑僅存在於歷史提交，不應重新引用。
- `api/start.py` 與 `api/__main__.py` 都包含 API／MindsDB 啟動編排；修改啟動行為時檢查兩者。它們引用的 `../../mindsdb/` 是本儲存庫外的獨立環境。

## 核心資料契約

新增或修改資料清理（DC）／特徵工程（FE）方法時，檢查並同步更新：

1. `automind/src/automind/models/preprocessing.py` 的 enum、recommendation schema 與 options。
2. `automind/src/automind/data_utils/template.py` 的提示詞與 `get_few_shot_prompt()` 範例。
3. `automind/src/automind/data_utils/preprocessing.py` 的實作與 `@register_method` 註冊。
4. `automind/src/automind/data_utils/logic_applier.py` 的建議分派、處理紀錄與輸出。
5. 對應測試與 [核心 README](automind/README.md) 的擴充說明。

- `models/shared.py` 的 `EnumByName` 以 enum **名稱**解析與序列化；方法 registry 也以 `method.name` 作索引。不要任意改名、改成數字值，或新增名稱衝突。
- 提示詞使用 `<json>...</json>`，解析流程依賴這個格式。調整 LLM 輸出格式時，同步檢查 schema、parser、提示詞與 `tests/data_utils/llm_response.txt` fixture。
- 保留 `LogicApplier` 對輸入 DataFrame 的複製與原始／處理後資料分離。前處理變更需驗證索引、目標欄位與特徵欄位的對齊。
- LLM 回應應透過既有 schema 與已註冊方法處理；不要引入直接執行回應文字的 `eval`／`exec`。

## API 與資料庫契約

- 修改 stored procedure 時同步檢查 Python parameter mapping；`exec_mutation_sp()` 的參數名稱必須與 SQL 參數一致。修改 view 時同步檢查 TypedDict 欄位，`get_view_by_id()` 會依 schema annotations 建立查詢。
- 保留既有回應中的 `state`、`message`、`new_id` 語意；HTTP request／response 變更需檢查 controller、model 與呼叫端。
- SQL 值使用綁定參數；動態 table、view、procedure 或欄位名稱必須來自受控映射或驗證，不直接插入使用者輸入。
- `HeaderSessionMiddleware` 由 `X-User-Id` 建立 request state；`env == "test"` 會使用固定身分。不要以切換正式設定到 test 模式來繞過驗證；部署 API 的金鑰驗證也須保留。

## 工作目錄與開發指令

指令須在指定套件目錄執行。套件與鎖檔是依賴來源；保留既有版本限制，不因文件或無關變更重鎖整組依賴。

| 工作目錄 | 指令 | 用途與條件 |
| --- | --- | --- |
| `automind/` 或 `api/` | `uv sync` | 安裝／同步該套件環境，會寫入本機環境且可能存取網路；不是唯讀檢查 |
| `automind/` | `uv run -m pytest ./src/automind/tests -x` | `pyproject.toml` 定義的核心測試指令；須先具備下節資料 |
| `automind/` | `uv run -m pytest ./src/automind/tests/data_utils/test_01_parser.py -x` | 欄位解析的聚焦測試，同樣依賴 dataset fixture |
| `api/` | `uv run start.py` | 啟動 API 與外部 MindsDB；須有設定、SQL Server、ODBC driver 與獨立 MindsDB 環境 |
| 儲存庫根目錄 | `git diff --check` | 檢查變更中的空白問題 |

- 根 README 中的 `./tests/data_utils` 不符合現有套件結構；測試採用上表的 `src/automind/tests` 路徑。
- `setup.ps1` 會替兩個套件建立環境並安裝依賴；不要為了查證文件執行它。`uv run` 也可能自動同步環境，執行前確認依賴變更在任務範圍內。
- 兩份 `pyproject.toml` 都有 Ruff 與 Pyright 設定。若開發環境已提供工具，在對應套件目錄執行 `ruff check src`、`pyright`；它們未列為專案依賴，不宣稱乾淨環境一定可執行。
- Python 使用既有型別註記與命名慣例；Ruff 行長為 80，忽略 `F403`、`E402`。避免順帶重排整份檔案或批次移除既有 Pyright suppression。

## 驗證方式與限制

- 核心 pytest fixtures 從 `automind/src/automind/data/csv/synthea_covid19_10k/` 讀取 `slice_patients.csv`，資料融合測試還讀取 `slice_conditions.csv`。這些資料未納入版本控制；執行前確認存在，不以空檔代替。
- `test_02_meta_generator.py` 會寫入同目錄的 `llm_query.txt`。LLM fixture 是本機文字檔，驗證前處理不需要實際呼叫 Azure。
- 修改單一功能時先跑對應測試；修改 schema、registry 或跨模組資料流程時，完成後跑完整核心測試。新增測試優先使用小型 DataFrame 與固定 LLM 回應，檢查資料契約、失敗行為與實際輸出。
- `api/src/automind_api/tests/` 包含連線、上傳、訓練與刪除操作的手動整合腳本，不能視為隔離的單元測試套件。新增 API 測試時 mock 資料庫、MindsDB 與 Azure 邊界，包括 lifespan 與匯入時的設定／client 初始化。
- 不在根目錄盲目執行無指定路徑的 pytest；實驗測試只從 `src/automind/tests` 明確收集。
- 專案未定義 CI 工作流程或 Markdown lint 指令。文件修改檢查路徑、連結、程式碼區塊與 `git diff --check` 即可，不為此啟動服務或訓練模型。
- 缺少依賴、資料、設定或工具時，明確回報未執行的檢查與原因；不得把靜態檢查描述為測試通過。

## 設定、機密與副作用

- API 設定讀取依賴目前工作目錄下的 `src/automind_api/configs/`，預設 `server.json`；Azure 使用該處的 `private.json`，且在模組匯入時讀取設定並建立 client。
- 核心 `utils/config.py` 使用 `Path().parent.absolute()` 尋找 `public.json`／`private.json`；這是相對目前工作目錄的運算，不能假設它相對模組位置。更改路徑邏輯前需檢查所有呼叫端。
- `.env`、`private.*`、server／MindsDB 設定與 SQL／batch 腳本可能包含機密或機器專用資訊。即使已被追蹤（例如 `automind/.env`），也不要把其值印到輸出、複製到文件、測試或新提交；查證時只讀必要結構並遮蔽敏感值。
- 未有使用者對相關操作的明確授權，不執行資料庫初始化、DDL、遠端新增／刪除、引擎上傳或真實 Azure 請求。沿用對話中已有的授權，不重複要求確認。
- `api/create-db/` 的 batch／SQL 會修改資料庫與帳號，部分 SQL 啟用 `TRUSTWORTHY`；需審閱實際目標與執行順序。`api/db-setup.bat` 仍引用不存在的 sqlpredictor 路徑，不把它列為可用初始化入口。
- 啟動 FastAPI 的 lifespan 會連接 SQL Server 與 MindsDB。不要以啟動應用作為純文件或無關變更的驗證。
- 實驗與 TPOT 程式可能進行耗時訓練、付費請求並覆寫 CSV、報表或 `tpot_best_pipeline.pkl`。測試使用 mock 與暫存目錄，不載入不受信任的 pickle。
- 不手動編輯 `.venv/`、`*.egg-info/`、cache 或工具產生的 typings。不要順帶重產實驗輸出、LLM 回應及報表，或替換 `api/create-db/i3s/` 的 DLL 二進位檔。
- 修改公開開發流程或核心擴充契約時更新對應 README；代理操作規則留在本文件。送出結果前確認只有任務相關變更，沒有新增機密、資料或模型產物。
