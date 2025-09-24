plots:
    wf[s]:
      type: multiline
      y_offset: 100
      var: ['wf_s:mcp', 'wf_s:u1', 'wf_s:u2', 'wf_s:v1', 'wf_s:v2', 'wf_s:w1', 'wf_s:w2']
      
    n[s]_rollavg:
      type: rollavg
      var: 'hit_s:n'
      window: {'w1':1000, 'w2':1000}

    scan_n[s]:
      type: scan_var
      var: 'hit_s:n'     
      decimals: 5
      scan: 'scan:var1'
      norm: 
         
    scan2_n[s]:
      type: scan2_var
      var: 'hit_s:n'     
      scan: ['scan:var2', 'scan:var3']
      decimal: [5, 5]
      norm: 'dummy'

    scan2_n[s]_func:
      type: scan2_var_func
      func: {name: range, args1: ['hit_s:n'], args2:[0, 100]}
      func_scan1: {name: test1, args1: ['scan:var1'], args2:[0, 100]}
      func_scan2: {name: test2, args1: ['scan:var2'], args2:[0, 100]}
      decimal: [5, 5]
      func_norm: {name: test3, args1: ['gmd'], args2:[0, 100]}


    scan_t[l]_func
      type: scan_hist1d_func
      func: {name: , args1: ['hit_l:t'], args2:}
      func_scan: {name:, args1: ['scan:var1']}
      decimal: 5
      norm:

    scan_t[s]:
      type: scan_hist1d
      arange: {'hit_s:t': [0, 10000, 10]}   
      scan: 'scan:var1'
      decimal: 5
      norm: 
      
    n[s]:
      type: hist1d
      arange: {'hit_s:n': [0, 10, 1]}      
      
    t[s]:
      type: hist1d
      arange: {'hit_s:t': [0, 10000, 1]}
    
    x-y[s]:
      type: hist2d
      arange: {'hit_s:x': [-60, 60, 1], 'hit_s:y': [-60, 60, 1]}
    
    x-t[s]:
      type: hist2d
      arange: {'hit_s:t': [0, 10000, 10], 'hit_s:x': [-60, 60, 1]}
    
    y-t[s]:
      type: hist2d
      arange: {'hit_s:t': [0, 10000, 10], 'hit_s:y': [-60, 60, 1]}

    atm: 
      type: single_line
      var: 'atm:line'

    atm_func:
      type: singleline_func
      func: {name: subtract, args1: ['atm:line'], args2:[100]}

    atm_avg:
      type: rollavg1d
      var: 'atm:line'
      window: 1000

    atm_avg_func:
      type: rollavg1d_func
      func: {name: subtract, args1: ['atm:line'], args2:[100]}
      window: 1000


    n[s]_rollavg_func:
      type: rollavg
      func: {name: range, args1:['hit_s:n'],args2:[0,10000]}
      window: {'w1':1000, 'w2':1000}

    atm_image: 
      type: single_image
      var: 'atm:opal'

    atm_sig_bkg:
      type: sigbkg1d
      plot_type: singleline_func 
      func_sig: {name: filter.atm, args1: ['atm:line', 'bld:xmgd', 'timing:280'], args2: [0.005, 1]}
      func_bkg: {name: filter.atm, args1: ['atm:line', 'bld:xgmd', 'timing:281'], args2: [0, 1]}

    lxt_scan_tof:
        type: sigbkg1d
        plot_type: hist1d_func
        func_sig: {name: filter.duck_goose, args1: ['tpks_e:1', 'n_tpks_e:1', 'timing:280'], args2:[1]}
        func_bkg: {name: filter.duck_goose, args1: ['tpks_e:1', 'n_tpks_e:1', 'timing:281'], args2:[1]}
        arange_var: [0, 10000, 10]
        norm_type: 'sum'
        func_norm_sig: {name: filter.duck_goose, args1: ['bld:xgmd', 'timing:280'], args2:[1]}
        func_norm_bkg: {name: filter.duck_goose, args1: ['bld:xgmd', 'timing:281'], args2:[1]}
