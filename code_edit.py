def conostEscan_show_table():
    # 新しいウィンドウを作成
    result_window = tk.Toplevel()
    result_window.title("calculation results (unit : deg)")
    
    # Treeviewの設定
    tree = ttk.Treeview(result_window, columns=("hw","h","k","l","C1", "A1", "C2", "A2", "C3", "A3", "mu", "nu"), show="headings")
    tree.pack(fill="both", expand=True)
    
    # 各列に見出しを設定
    for col in tree["columns"]:
        tree.heading(col, text=col)
        tree.column(col, width=80, anchor="center")
    
    # RLtableを取得し、辞書から必要な変数を取り出す
    RLtable = on_Rlcalc()
    astar = RLtable['astar']
    bstar = RLtable['bstar']
    cstar = RLtable['cstar']
    
    # 先にフロアの条件を読み込み
    # INIファイルから設定を読み込む
    config = configparser.ConfigParser()
    # .exe化した場合に対応する
    if getattr(sys, 'frozen', False):
        # .exeの場合、sys.argv[0]が実行ファイルのパスになる
        ini_path = os.path.join(os.path.dirname(sys.argv[0]), 'config.ini')
    else:
        # .pyの場合、__file__がスクリプトのパスになる
        ini_path = os.path.join(os.path.dirname(__file__), 'config.ini')
    config.read(ini_path)
    
    monochromator_radius = float(config['settings']['monochromator_radius'])
    monochromator_to_sample = float(config['settings']['monochromator_to_sample'])
    sample_goniometer_radius = float(config['settings']['sample_goniometer_radius'])
    sample_to_analyzer = float(config['settings']['sample_to_analyzer'])
    analyzer_radius = float(config['settings']['analyzer_radius'])
    analyzer_to_detector = float(config['settings']['analyzer_to_detector'])
    detector_radius = float(config['settings']['detector_radius'])
    floor_length = float(config['settings']['floor_length'])
    floor_width = float(config['settings']['floor_width'])
    floor_position_x = float(config['settings']['floor_position_x'])
    floor_position_y = float(config['settings']['floor_position_y'])

    # Ei or Ef fixの判定
    fixe=float(eief.get())
    
    global angletable3
    angletable3 = angle_calc3(astar,bstar,cstar,U,B,UB,bpe,bpc2,bpmu,bpnu,bp,fixe,hw_cal,h_ini,k_ini,l_ini,h_fin,k_fin,l_fin,h_inc,k_inc,l_inc)
    
    A_sets = []  # A_setsリストを初期化
    QE_sets = []
    C_sets = []
    # resultsリストの各結果をTreeviewに追加
    for results in angletable3:
        values = tuple(results.values())
        item_id = tree.insert("", "end", values=values)
    
        # A1, A2, A3 を取得して A_sets に追加
        A1 = round(results['A1'], 4)  # 'A1'
        A2 = -round(results['A2'], 4)  # 'A2'
        A3 = round(results['A3'], 4)  # 'A3'
        A_sets.append([A1, A2, A3])  # A_sets に追加
        C1 = round(results['C1'], 4)  # 'C1'
        C2 = round(results['C2'], 4)  # 'C2'
        C3 = round(results['C3'], 4)  # 'C3'
        C4 = round(results['offset'], 4)  # 'offset'
        C_sets.append([C1, C2, C3, C4])
        # hw, h,k,l
        hw = round(results['hw'], 4)  # 'A1'
        h = round(results['h'], 4)  # 'A2'
        k = round(results['k'], 4)  # 'A3'
        l = round(results['l'], 4)  # 'A3'
        QE_sets.append([hw, h, k,l])
        
        # sample gonioがfloorからはみ出る場合
        positionY_sample = monochromator_to_sample * np.sin(np.radians(results['A1'])) - sample_goniometer_radius # < floor_position_y
        # analyzer dramがfloorからはみ出る場合
        positionY_analyzer = monochromator_to_sample * np.sin(np.radians(results['A1'])) - sample_to_analyzer * np.sin(np.radians(results['A2'] - results['A1'])) - analyzer_radius # < floor_position_y
        # detector dramがfloorからはみ出る場合
        positionY_detector = monochromator_to_sample * np.sin(np.radians(results['A1'])) - sample_to_analyzer * np.sin(np.radians(results['A2'] - results['A1'])) + analyzer_to_detector * np.sin(np.radians(results['A3'] - results['A2'] + results['A1'])) - detector_radius # < floor_position_y
        positionX_detector = monochromator_to_sample * np.cos(np.radians(results['A1'])) + sample_to_analyzer * np.cos(np.radians(results['A2'] - results['A1'])) + analyzer_to_detector * np.cos(np.radians(results['A3'] - results['A2'] + results['A1'])) - detector_radius # < floor_position_x+floor_length
        
        # フロアからはみ出た場合、行を赤色にする
        if (positionY_sample < floor_position_y or
          positionY_analyzer < floor_position_y or
          positionY_detector < floor_position_y or
          positionX_detector > floor_position_x+floor_length):
            tree.tag_configure("red", foreground="red")  # 'red' タグを設定
            tree.item(item_id, tags=("red",))  # 行に 'red' タグを適用

        # hardware limitを超えた場合、行をい色にする
        elif (round(results['C1'],4)<float(hwl2f.get()) or
            round(results['C1'],4)>float(hwl2t.get()) or
            round(results['A1'],4)<float(hwl3f.get()) or
            round(results['A1'],4)>float(hwl3t.get()) or
            round(results['C2'],4)<float(hwl4f.get()) or
            round(results['C2'],4)>float(hwl4t.get()) or
            round(results['A2'],4)<float(hwl5f.get()) or
            round(results['A2'],4)>float(hwl5t.get()) or
            round(results['C3'],4)<float(hwl6f.get()) or
            round(results['C3'],4)>float(hwl6t.get()) or
            round(results['A3'],4)<float(hwl7f.get()) or
            round(results['A3'],4)>float(hwl7t.get()) or
            round(results['mu'],4)<float(hwl8f.get()) or
            round(results['mu'],4)>float(hwl8t.get()) or
            round(results['nu'],4)<float(hwl9f.get()) or
            round(results['nu'],4)>float(hwl9t.get())):
            tree.tag_configure("blue", foreground="blue")  # 'red' タグを設定
            tree.item(item_id, tags=("blue",))  # 行に 'red' タグを適用
        