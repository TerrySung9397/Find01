import os
import re
import json
import urllib.request
import ssl
import pandas as pd
import numpy as np
from datetime import datetime

# Try importing yfinance, if not installed, we will prompt the user to install it
try:
    import yfinance as yf
except ImportError:
    print("[!] 偵測到未安裝 yfinance。請先執行: pip install yfinance")
    import sys
    sys.exit(1)

def get_taiwan_stock_list():
    """
    Fetch the list of all Listed (上市) and OTC (上櫃) stocks from TWSE ISIN website.
    Filters out non-common stocks (only CFICode starting with 'ES' for common stocks).
    """
    print("[*] 正在從證交所 ISIN 網站獲取台灣股市股票清單...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    stocks = {}
    
    # strMode=2 is Listed (上市), strMode=4 is OTC (上櫃)
    modes = [("2", "上市", ".TW"), ("4", "上櫃", ".TWO")]
    
    for str_mode, market_name, suffix in modes:
        url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={str_mode}"
        req = urllib.request.Request(url, headers=headers)
        try:
            print(f"[*] 獲取 {market_name} 股票清單中...")
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(req, timeout=20, context=context) as response:
                html = response.read().decode('cp950', errors='ignore')
                
            td_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL)
            tr_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL)
            
            rows = tr_pattern.findall(html)
            count = 0
            for row in rows:
                tds = td_pattern.findall(row)
                if len(tds) >= 6:
                    # Clean HTML tags inside td elements
                    cell0 = re.sub(r'<[^>]+>', '', tds[0]).strip()  # e.g., "2330　台積電"
                    cell3 = re.sub(r'<[^>]+>', '', tds[3]).strip()  # 市場屬性 (e.g. 上市/上櫃)
                    cell4 = re.sub(r'<[^>]+>', '', tds[4]).strip()  # 產業分組 (e.g. 半導體業)
                    cfi = re.sub(r'<[^>]+>', '', tds[5]).strip()    # CFICode
                    
                    # CFICode starts with 'ES' indicates standard Common Stocks (普通股)
                    if cfi.startswith('ES'):
                        parts = re.split(r'[\s\u3000]+', cell0, maxsplit=1)
                        if len(parts) == 2:
                            code, name = parts
                            if len(code) == 4 and code.isdigit():
                                stocks[code] = {
                                    "code": code,
                                    "name": name,
                                    "market": market_name,
                                    "industry": cell4,
                                    "ticker": f"{code}{suffix}"
                                }
                                count += 1
            print(f"[+] 成功解析出 {count} 檔 {market_name} 普通股")
        except Exception as e:
            print(f"[x] 獲取 {market_name} 股票清單時發生錯誤: {e}")
            
    print(f"[+] 共計獲取 {len(stocks)} 檔台灣股市普通股")
    return stocks

def calculate_macd(df, short=6, long=13, signal=9):
    """
    Calculate MACD with parameters (short, long, signal).
    By default: short=6, long=13, signal=9 as requested.
    """
    close = df['Close']
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()
    dif = ema_short - ema_long
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = dif - dea
    return dif, dea, macd_hist

def calculate_kdj(df, n=9, m1=3, m2=3):
    """
    Calculate KDJ (9, 3, 3).
    RSV = (Close - Low_9) / (High_9 - Low_9) * 100
    K = 2/3 * K_prev + 1/3 * RSV
    D = 2/3 * D_prev + 1/3 * K
    J = 3 * K - 2 * D
    """
    close = df['Close']
    low_n = df['Low'].rolling(window=n).min()
    high_n = df['High'].rolling(window=n).max()
    
    # Handle division by zero when high_n == low_n (limit up/down with flat price)
    diff = high_n - low_n
    rsv = (close - low_n) / diff * 100
    rsv = rsv.fillna(50.0)
    
    k_list = []
    d_list = []
    
    k_val = 50.0
    d_val = 50.0
    
    for val in rsv:
        if pd.isna(val):
            k_val = 50.0
            d_val = 50.0
        else:
            k_val = (2.0 / 3.0) * k_val + (1.0 / 3.0) * val
            d_val = (2.0 / 3.0) * d_val + (1.0 / 3.0) * k_val
        k_list.append(k_val)
        d_list.append(d_val)
        
    k_series = pd.Series(k_list, index=df.index)
    d_series = pd.Series(d_list, index=df.index)
    j_series = 3.0 * k_series - 2.0 * d_series
    
    return k_series, d_series, j_series

def fetch_regular_volumes_mis(stocks):
    """
    Fetch regular + after-hours fixed-price trading volumes (普通整股交易量) for all candidate stocks
    directly from TWSE MIS API.
    Returns a dict { 'code': volume_in_shares }
    """
    volumes = {}
    context = ssl._create_unverified_context()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    codes = list(stocks.keys())
    batch_size = 100
    print(f"\n[*] 正在向證交所 MIS 系統批次查詢 {len(codes)} 檔候選股票之官方『普通整股交易量』...")
    
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        ex_ch_list = []
        for code in batch:
            market = stocks[code]['market']
            prefix = "tse" if market == "上市" else "otc"
            ex_ch_list.append(f"{prefix}_{code}.tw")
            
        ex_ch = "|".join(ex_ch_list)
        url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch={ex_ch}"
        
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=context, timeout=10) as r:
                res_data = json.loads(r.read().decode('utf-8'))
                msg_array = res_data.get('msgArray', [])
                for item in msg_array:
                    code = item.get('c')
                    v_str = item.get('v')
                    fv_str = item.get('fv', '0')
                    if code and v_str:
                        try:
                            # v and fv in MIS API are already in lots (張), convert to shares (股)
                            vol_lots = int(v_str) + int(fv_str)
                            volumes[code] = vol_lots * 1000
                        except ValueError:
                            pass
        except Exception as e:
            # We don't print too many batch errors to keep the console clean
            pass
            
    print(f"[+] 成功獲取 {len(volumes)} 檔候選股票之官方『普通整股交易量』(排除零股與鉅額交易)")
    return volumes

def screen_stocks(stocks, min_price=None, max_price=None, min_volume=None, regular_volumes=None):
    """
    Download daily stock data in batches and screen for:
    1. Latest Close price within range [min_price, max_price]
    2. Latest Volume >= min_volume (using official regular trading volume if available)
    3. MACD Histogram < 0 (Green bar)
    4. KDJ J-line pointing upwards (J_today > J_yesterday)
    """
    tickers = list(stocks.keys())
    if regular_volumes is None:
        regular_volumes = {}
    yf_tickers = [stocks[code]['ticker'] for code in tickers]
    
    chunk_size = 100
    matched_stocks = []
    total_scanned = 0
    total_download_errors = 0
    
    print(f"\n[*] 開始批次下載歷史股價並計算技術指標 (每批次 {chunk_size} 檔)...")
    
    # Loop over tickers in chunks
    for i in range(0, len(yf_tickers), chunk_size):
        chunk = yf_tickers[i:i+chunk_size]
        chunk_str = " ".join(chunk)
        batch_num = i // chunk_size + 1
        total_batches = (len(yf_tickers) - 1) // chunk_size + 1
        
        print(f"[*] 進度: 批次 {batch_num}/{total_batches} ({len(chunk)} 檔股票)...")
        
        try:
            # Fetch 3 months of daily price data to ensure MACD and KDJ stabilize
            df = yf.download(chunk_str, period="3mo", group_by="ticker", progress=False, threads=True)
            
            if df.empty:
                print(f"[!] 批次 {batch_num} 下載數據為空，跳過。")
                continue
                
            # Check if columns is MultiIndex (multiple tickers download)
            if isinstance(df.columns, pd.MultiIndex):
                present_tickers = df.columns.get_level_values(0).unique()
                for ticker in present_tickers:
                    ticker_df = df[ticker].copy()
                    ticker_df = ticker_df.dropna(subset=['Close'])
                    
                    # We need at least 15 trading days to calculate moving averages properly
                    if len(ticker_df) < 15:
                        continue
                        
                    total_scanned += 1
                    
                    # Calculate MACD(6, 13, 9)
                    dif, dea, macd_hist = calculate_macd(ticker_df, short=6, long=13, signal=9)
                    # Calculate KDJ(9, 3, 3)
                    k_s, d_s, j_s = calculate_kdj(ticker_df, n=9, m1=3, m2=3)
                    
                    # Get the values for latest and previous days
                    latest_close = ticker_df['Close'].iloc[-1]
                    latest_volume = ticker_df['Volume'].iloc[-1]
                    macd_val = macd_hist.iloc[-1]
                    macd_val_prev = macd_hist.iloc[-2]
                    macd_val_prev2 = macd_hist.iloc[-3]
                    
                    dif_val = dif.iloc[-1]
                    dea_val = dea.iloc[-1]
                    k_val = k_s.iloc[-1]
                    d_val = d_s.iloc[-1]
                    j_val = j_s.iloc[-1]
                    
                    k_prev = k_s.iloc[-2]
                    d_prev = d_s.iloc[-2]
                    j_prev = j_s.iloc[-2]
                    
                    # Target condition check:
                    # 1. Price within range: min_price <= latest_close <= max_price
                    price_ok = True
                    if min_price is not None and latest_close < min_price:
                        price_ok = False
                    if max_price is not None and latest_close > max_price:
                        price_ok = False
                        
                    # 2. Volume filter (Volume is in shares, min_volume is in shares)
                    code = ticker.split('.')[0]
                    actual_volume = regular_volumes.get(code, latest_volume)
                    
                    volume_ok = True
                    if min_volume is not None and actual_volume < min_volume:
                        volume_ok = False
                        
                    # 3. MACD green column shortening for 2 consecutive days AND DIF > -0.05:
                    macd_ok = (
                        macd_val < 0 and 
                        macd_val_prev < 0 and 
                        macd_val_prev2 < 0 and 
                        macd_val > macd_val_prev and 
                        macd_val_prev > macd_val_prev2 and
                        dif_val > -0.05
                    )
                    
                    # 4. KDJ J-line strongly golden crosses K and D from low level:
                    kdj_ok = (
                        j_prev < k_prev and 
                        j_prev < d_prev and 
                        j_val > k_val and 
                        j_val > d_val and 
                        j_prev < 30 and 
                        j_val >= 20
                    )
                    
                    if price_ok and volume_ok and macd_ok and kdj_ok:
                        info = stocks.get(code, {})
                        
                        matched_stocks.append({
                            "code": code,
                            "name": info.get("name", "未知"),
                            "market": info.get("market", "未知"),
                            "industry": info.get("industry", "未知"),
                            "close": latest_close,
                            "volume": actual_volume,
                            "macd_hist": macd_val,
                            "dif": dif_val,
                            "dea": dea_val,
                            "k": k_val,
                            "d": d_val,
                            "j": j_val,
                            "j_prev": j_prev,
                            "j_change": j_val - j_prev
                        })
            else:
                # Single ticker fallback
                ticker = chunk[0]
                ticker_df = df.copy()
                ticker_df = ticker_df.dropna(subset=['Close'])
                if len(ticker_df) >= 15:
                    total_scanned += 1
                    dif, dea, macd_hist = calculate_macd(ticker_df, short=6, long=13, signal=9)
                    k_s, d_s, j_s = calculate_kdj(ticker_df, n=9, m1=3, m2=3)
                    
                    # Get the values for latest and previous days
                    latest_close = ticker_df['Close'].iloc[-1]
                    latest_volume = ticker_df['Volume'].iloc[-1]
                    macd_val = macd_hist.iloc[-1]
                    macd_val_prev = macd_hist.iloc[-2]
                    macd_val_prev2 = macd_hist.iloc[-3]
                    
                    dif_val = dif.iloc[-1]
                    dea_val = dea.iloc[-1]
                    k_val = k_s.iloc[-1]
                    d_val = d_s.iloc[-1]
                    j_val = j_s.iloc[-1]
                    
                    k_prev = k_s.iloc[-2]
                    d_prev = d_s.iloc[-2]
                    j_prev = j_s.iloc[-2]
                    
                    price_ok = True
                    if min_price is not None and latest_close < min_price:
                        price_ok = False
                    if max_price is not None and latest_close > max_price:
                        price_ok = False
                        
                    # 2. Volume filter (Volume is in shares, min_volume is in shares)
                    code = ticker.split('.')[0]
                    actual_volume = regular_volumes.get(code, latest_volume)
                    
                    volume_ok = True
                    if min_volume is not None and actual_volume < min_volume:
                        volume_ok = False
                        
                    # 3. MACD green column shortening for 2 consecutive days AND DIF > -0.05:
                    macd_ok = (
                        macd_val < 0 and 
                        macd_val_prev < 0 and 
                        macd_val_prev2 < 0 and 
                        macd_val > macd_val_prev and 
                        macd_val_prev > macd_val_prev2 and
                        dif_val > -0.05
                    )
                    
                    # 4. KDJ J-line strongly golden crosses K and D from low level:
                    kdj_ok = (
                        j_prev < k_prev and 
                        j_prev < d_prev and 
                        j_val > k_val and 
                        j_val > d_val and 
                        j_prev < 30 and 
                        j_val >= 20
                    )
                    
                    if price_ok and volume_ok and macd_ok and kdj_ok:
                        info = stocks.get(code, {})
                        matched_stocks.append({
                            "code": code,
                            "name": info.get("name", "未知"),
                            "market": info.get("market", "未知"),
                            "industry": info.get("industry", "未知"),
                            "close": latest_close,
                            "volume": actual_volume,
                            "macd_hist": macd_val,
                            "dif": dif_val,
                            "dea": dea_val,
                            "k": k_val,
                            "d": d_val,
                            "j": j_val,
                            "j_prev": j_prev,
                            "j_change": j_val - j_prev
                        })
                        
        except Exception as e:
            print(f"[!] 下載或計算此批次時發生異常: {e}")
            total_download_errors += len(chunk)
            
    print(f"\n[+] 篩選完成！共掃描 {total_scanned} 檔股票，排除失敗/停牌個股。")
    price_cond_str = ""
    if min_price is not None or max_price is not None:
        min_p = min_price if min_price is not None else 0
        max_p = max_price if max_price is not None else "無上限"
        price_cond_str = f"且股價在 {min_p} ~ {max_p} 元之間"
    vol_cond_str = ""
    if min_volume is not None:
        vol_cond_str = f"且成交量大於 {min_volume / 1000:,.0f} 張"
    print(f"[+] 符合條件「MACD綠柱連2天縮短且DIF > -0.05 (6,13,9) 且 KDJ 的 J 線低檔強勢黃金交叉 K/D 線且 J >= 20 (9,3,3){price_cond_str}{vol_cond_str}」的股票共計: {len(matched_stocks)} 檔。")
    return matched_stocks

def send_telegram_report(matched_stocks, min_price=None, max_price=None, industry_filter=None, min_volume=None, token=None, chat_id=None):
    """
    Format and send stock screening results to Telegram via bot.
    """
    if not token or not chat_id:
        print("[!] 缺少 Telegram Token 或 Chat ID，跳過發送。")
        return
        
    print(f"\n[*] 正在準備發送結果至 Telegram (Chat ID: {chat_id})...")
    
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Construct HTML message
    lines = []
    lines.append("📊 <b>台灣股市選股篩選結果</b>")
    lines.append(f"📅 <b>篩選時間</b>: <code>{now_str}</code>")
    lines.append("")
    lines.append("💡 <b>篩選條件說明</b>:")
    lines.append("1. MACD 綠柱連2天縮短且 DIF &gt; -0.05 (6,13,9)")
    lines.append("2. KDJ J線低檔強勢黃金交叉 K/D 線，且 J &gt;= 20 (9,3,3)")
    
    conds = []
    if min_price is not None or max_price is not None:
        min_p = min_price if min_price is not None else 0
        max_p = max_price if max_price is not None else "無上限"
        conds.append(f"股價 {min_p}~{max_p} 元")
    if min_volume is not None:
        conds.append(f"普通整股成交量 &gt;= {min_volume/1000:,.0f} 張")
    if industry_filter is not None:
        conds.append(f"指定產業: {industry_filter}")
        
    if conds:
        lines.append(f"3. 額外限制: " + ", ".join(conds))
        
    lines.append("")
    lines.append(f"✅ <b>符合條件個股</b>: <b>{len(matched_stocks)}</b> 檔")
    lines.append("="*20)
    
    if len(matched_stocks) == 0:
        lines.append("😢 今日無符合條件之股票。")
    else:
        for idx, s in enumerate(matched_stocks):
            vol_lots = s['volume'] / 1000
            # Bold stock name and code
            lines.append(
                f"<b>{idx+1}. {s['name']} ({s['code']})</b>\n"
                f"   • 收盤價: <b>{s['close']:.2f} 元</b>\n"
                f"   • 當日成交量: <b>{vol_lots:,.0f} 張</b> (普通整股)\n"
                f"   • KDJ J值: <b>{s['j']:.1f}</b> (增幅 +{s['j_change']:.1f})\n"
                f"   • MACD柱值: <code>{s['macd_hist']:.2f}</code> | DIF: <code>{s['dif']:.2f}</code>"
            )
            lines.append("-" * 20)
            
    # Remove the last separator if it exists
    if len(matched_stocks) > 0 and lines[-1] == "-" * 20:
        lines.pop()
        
    message = "\n".join(lines)
    
    # Send HTTP POST request
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=context, timeout=15) as response:
            res_content = response.read().decode('utf-8')
            res_json = json.loads(res_content)
            if res_json.get("ok"):
                print("[+] Telegram 訊息推播成功！")
                return True
            else:
                print(f"[x] Telegram API 回傳錯誤: {res_content}")
                return False
    except Exception as e:
        print(f"[x] 發送 Telegram 訊息時發生異常: {e}")
        return False

def generate_report(matched_stocks, match_limit=None, min_price=None, max_price=None, industry_filter=None, min_volume=None):

    """
    Generate output table and write results to markdown artifact file.
    """
    # Sort matched stocks by code
    matched_stocks = sorted(matched_stocks, key=lambda x: x['code'])
    
    # Apply match limit if specified
    original_match_count = len(matched_stocks)
    if match_limit is not None and match_limit > 0:
        matched_stocks = matched_stocks[:match_limit]
        print(f"[*] 已限制輸出前 {match_limit} 檔符合條件的股票（原始符合 {original_match_count} 檔）。")
    
    # Construct Markdown Report
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = []
    report.append("# Taiwan Stock Screening Results")
    report.append(f"\n**篩選時間**: `{now_str}`")
    report.append("\n## 篩選條件說明")
    report.append("1. **MACD 綠柱連續 2 天縮短且 DIF > -0.05 (6, 13, 9)**: 連續三日皆為綠柱（柱值 < 0），今日柱值大於昨日，昨日柱值大於前日；且當前 DIF 快線大於 -0.05（代表中線偏強）。")
    report.append("2. **KDJ J線低檔強勢黃金交叉 K/D 線 (9, 3, 3)**: 昨日 J 值小於 K 與 D，今日 J 值大於 K 與 D，且昨日 J 值處於低檔（< 30），今日 J 值大於等於 20。")
    if min_price is not None or max_price is not None:
        min_p = min_price if min_price is not None else 0
        max_p = max_price if max_price is not None else "無上限"
        report.append(f"3. **股價區間**: 股價必須介於 `{min_p}` 至 `{max_p}` 元之間。")
    if industry_filter is not None:
        report.append(f"4. **指定產業**: 僅包含 `{industry_filter}` 相關產業。")
    if min_volume is not None:
        report.append(f"5. **成交量限制**: 最新一日交易量必須大於等於 `{min_volume/1000:,.0f}` 張（{min_volume:,} 股）。")
        
    if match_limit is not None and match_limit > 0:
        report.append(f"\n**篩選結果**: 顯示前 `{len(matched_stocks)}` 檔股票（共計符合 `{original_match_count}` 檔）")
    else:
        report.append(f"\n**符合條件之股票總數**: `{len(matched_stocks)}` 檔")
    report.append("\n---\n")
    report.append("## 符合條件之股票清單")
    report.append("\n| 股號 | 股名 | 市場 | 產業分類 | 收盤價 (TWD) | 當日成交量 (張) | MACD 柱值 | DIF / DEA | K / D 值 | 當前 J 值 | J 線增幅 (ΔJ) |")
    report.append("|---|---|---|---|---|---|---|---|---|---|---|")
    
    for s in matched_stocks:
        dif_dea_str = f"{s['dif']:.2f} / {s['dea']:.2f}"
        kd_str = f"{s['k']:.1f} / {s['d']:.1f}"
        vol_lots = s['volume'] / 1000
        report.append(
            f"| `{s['code']}` | {s['name']} | {s['market']} | {s['industry']} | **{s['close']:.2f}** | "
            f"**{vol_lots:,.0f} 張** | `{s['macd_hist']:.2f}` | {dif_dea_str} | {kd_str} | **{s['j']:.1f}** | +{s['j_change']:.1f} |"
        )
        
    report_content = "\n".join(report)
    
    # Write to local markdown file
    output_path = r"d:\TPA\screener_results.md"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"[+] 篩選結果成功寫入至: {output_path}")
    except Exception as e:
        print(f"[x] 寫入結果檔案失敗: {e}")
        
    # Also print standard output table for console
    print("\n" + "="*90)
    if match_limit is not None and match_limit > 0:
        print(f" 符合條件之股票清單 (顯示前 {len(matched_stocks)} 檔，共 {original_match_count} 檔) ")
    else:
        print(f" 符合條件之股票清單 (共 {len(matched_stocks)} 檔) ")
    print("="*90)
    print(f"{'股號':<6}{'股名':<8}{'市場':<6}{'收盤':<8}{'成交量(張)':<12}{'MACD柱':<10}{'K/D值':<12}{'J值':<8}{'J增幅':<6}")
    print("-"*100)
    for s in matched_stocks[:50]:  # Limit console print to first 50 to avoid clutter
        kd_str = f"{s['k']:.1f}/{s['d']:.1f}"
        vol_str = f"{s['volume']/1000:,.0f}張"
        print(f"{s['code']:<6}{s['name']:<8}{s['market']:<6}{s['close']:<8.2f}{vol_str:<12}{s['macd_hist']:<10.2f}{kd_str:<12}{s['j']:<8.1f}+{s['j_change']:<6.1f}")
    
    if len(matched_stocks) > 50:
        print(f"\n... 還有 {len(matched_stocks) - 50} 檔股票，完整名單請查看 {output_path} ...")
    print("="*90)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="台灣股市 MACD & KDJ 技術指標篩選器")
    parser.add_argument(
        "--scan-limit", "-s", 
        type=int, 
        default=None, 
        help="限制要掃描的股票數量（例如：-s 50，適合快速測試，預設為掃描全部）"
    )
    parser.add_argument(
        "--match-limit", "-m", 
        type=int, 
        default=None, 
        help="限制輸出的符合條件股票數量上限"
    )
    parser.add_argument(
        "--min-price", "--min",
        type=float,
        default=None,
        help="限制股價最低價格（例如：--min-price 30）"
    )
    parser.add_argument(
        "--max-price", "--max",
        type=float,
        default=None,
        help="限制股價最高價格（例如：--max-price 150）"
    )
    parser.add_argument(
        "--industry", "-i",
        type=str,
        default=None,
        help="限制股市產業類別，多個以逗號分隔（例如：半導體,光電,電腦週邊）"
    )
    parser.add_argument(
        "--min-volume", "--vol",
        type=int,
        default=3000,
        help="限制交易量最低張數，單位為張（例如：--min-volume 3000，預設為 3000 張）"
    )
    parser.add_argument(
        "--tg-token",
        type=str,
        default="8812419373:AAE3E5f7dBH40JmPbn7h91JzsxJfZv2tdgw",
        help="Telegram Bot Token"
    )
    parser.add_argument(
        "--tg-chat-id",
        type=str,
        default="-5179213819",
        help="Telegram Chat ID"
    )
    parser.add_argument(
        "--tg",
        action="store_true",
        help="啟用 Telegram 訊息發送"
    )
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    # 1. Fetch all stock symbols
    stocks = get_taiwan_stock_list()
    
    if not stocks:
        print("[x] 無法獲取股票清單，程式終止。")
    else:
        # Limit by industry if specified (Perform early-filtering to save download time and bandwidth!)
        if args.industry is not None:
            raw_industries = [ind.strip() for ind in args.industry.split(',')]
            target_industries = []
            for ind in raw_industries:
                # 對常見的使用者友善名稱進行映射，以確保與證交所官方類別精確匹配
                term = ind
                if "通訊" in ind:
                    term = "通信"
                elif "電腦週邊" in ind:
                    term = "電腦及週邊"
                target_industries.append(term)
            filtered_stocks = {}
            for code, info in stocks.items():
                stock_ind = info.get("industry", "")
                match = False
                for target_ind in target_industries:
                    if target_ind in stock_ind:
                        match = True
                        break
                if match:
                    filtered_stocks[code] = info
            stocks = filtered_stocks
            print(f"[*] 產業篩選已套用，僅保留指定產業：{args.industry}（共 {len(stocks)} 檔股票）。")
            
        # Limit scanning count if specified
        if args.scan_limit is not None and args.scan_limit > 0:
            # Keep only the first scan_limit stocks
            stocks = {k: v for i, (k, v) in enumerate(stocks.items()) if i < args.scan_limit}
            print(f"[*] 已限制僅掃描前 {len(stocks)} 檔股票。")
            
        # Convert lots to shares (1 lot = 1,000 shares)
        min_volume_shares = args.min_volume * 1000 if args.min_volume is not None else None
        
        # Find the latest trading date using a quick download of one active stock
        print("[*] 正在向 Yahoo Finance 查詢最新交易日期...")
        latest_date_str = datetime.now().strftime("%Y%m%d")
        try:
            test_df = yf.download("2330.TW", period="5d", progress=False)
            if not test_df.empty:
                latest_date = test_df.index[-1]
                latest_date_str = latest_date.strftime("%Y%m%d")
                print(f"[+] 最新交易日期為: {latest_date_str}")
        except Exception as e:
            print(f"[!] 無法使用 yfinance 查詢最新日期，將預設使用當前日期: {e}")
            
        # Fetch official regular trading volumes from TWSE & TPEx
        regular_volumes = fetch_regular_volumes_mis(stocks)
        
        # 2. Run stock screening
        matched_stocks = screen_stocks(
            stocks, 
            min_price=args.min_price, 
            max_price=args.max_price, 
            min_volume=min_volume_shares,
            regular_volumes=regular_volumes
        )
        
        # 3. Output results
        generate_report(
            matched_stocks, 
            match_limit=args.match_limit, 
            min_price=args.min_price, 
            max_price=args.max_price, 
            industry_filter=args.industry,
            min_volume=min_volume_shares
        )
        
        # 4. Send Telegram Report
        if args.tg:
            send_telegram_report(
                matched_stocks,
                min_price=args.min_price,
                max_price=args.max_price,
                industry_filter=args.industry,
                min_volume=min_volume_shares,
                token=args.tg_token,
                chat_id=args.tg_chat_id
            )
        
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n[*] 總花費時間: {duration.total_seconds():.1f} 秒")
