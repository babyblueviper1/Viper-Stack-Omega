def analysis_pass(user_input, strategy, threshold, dest_addr, dao_percent, future_multiplier):
    global pruned_utxos_global, input_vb_global, output_vb_global

    addr = user_input.strip()
    is_xpub = addr.startswith(('xpub', 'zpub', 'ypub', 'tpub'))

    if is_xpub:
        utxos, msg = fetch_all_utxos_from_xpub(addr, threshold)
        if not utxos:
            return msg or "Failed to scan xpub", gr.update(visible=False)
    else:
        if not addr:
            return "Enter address or xpub", gr.update(visible=False)
        utxos = get_utxos(addr, threshold)
        if not utxos:
            return (
                "<div style='text-align:center;padding:30px;color:#f7931a;font-size:1.4rem;'>"
                "No UTXOs found above dust threshold<br><br>"
                "Try a different address or lower the dust limit"
                "</div>",
                gr.update(visible=False),
                gr.update(visible=False),
                "",
                []
            )

    utxos.sort(key=lambda x: x['value'], reverse=True)

    # Detect script type
    sample = [u.get('address') or addr for u in utxos[:10]]
    types = [address_to_script_pubkey(a)[1]['type'] for a in sample]
    from collections import Counter
    detected = Counter(types).most_common(1)[0][0] if types else "SegWit"

    vb_map = {
        'P2PKH': (148, 34),
        'P2SH': (91, 32),
        'SegWit': (68, 31),
        'Taproot': (57, 43)
    }
    input_vb_global, output_vb_global = vb_map.get(detected.split()[0], (68, 31))

    # === STRATEGY MAPPING ===
    NUCLEAR_OPTION = "NUCLEAR PRUNE (90% sacrificed — for the brave)"
    ratio_map = {
        "Privacy First (30% pruned)": 0.3,
        "Recommended (40% pruned)": 0.4,
        "More Savings (50% pruned)": 0.5,
        NUCLEAR_OPTION: 0.9,
    }
    ratio = ratio_map.get(strategy, 0.4)

    name_map = {
        "Privacy First (30% pruned)": "Privacy First",
        "Recommended (40% pruned)": "Recommended",
        "More Savings (50% pruned)": "More Savings",
        NUCLEAR_OPTION: '<span style="color:#ff1361; font-weight:900; text-shadow: 0 0 10px #ff0066;">NUCLEAR PRUNE</span>',
    }
    strategy_name = name_map.get(strategy, strategy.split(" (")[0])

    # === NUCLEAR MODE ===
    if strategy == NUCLEAR_OPTION:
        keep = min(3, len(utxos)) if len(utxos) > 0 else 0
        keep = max(1, keep)
    else:
        keep = max(1, int(len(utxos) * (1 - ratio)))

    pruned_utxos_global = utxos[:keep]

    nuclear_warning = ""
    if strategy == NUCLEAR_OPTION:
        nuclear_warning = '<br><span style="color:#ff0066; font-weight:bold; font-size:18px;">NUCLEAR MODE ACTIVE<br>Only the strongest survive.</span>'

    # ==================================================================
    # TEMPORARY RED TEST TABLE — REMOVE THIS WHEN YOU SEE IT WORKING
    # ==================================================================
    table_html = textwrap.dedent("""\
        <div style="border:4px solid red; padding:30px; background:#200; margin:20px 0; border-radius:16px; text-align:center;">
            <h3 style="color:#f7931a; font-size:28px; margin:0 0 20px 0;">
                TEST TABLE — IF YOU SEE THIS RED BOX, EVERYTHING IS WORKING
            </h3>
            <p style="color:white; font-size:22px;">Wiring = Fixed<br>Indentation = Fixed<br>Coin Control = Ready</p>
            <div style="background:#000; color:#f7931a; padding:20px; border-radius:12px; margin-top:20px; font-family:monospace;">
                Found: {total_utxos} UTXOs → Keeping: {keep} largest<br>
                Strategy: {strategy_name}
            </div>
        </div>""".format(total_utxos=len(utxos), keep=keep, strategy_name=strategy_name))

    # ==================================================================
    # WHEN THE RED BOX APPEARS → UNCOMMENT THE REAL TABLE BELOW
    # ==================================================================
    """
    # REAL COIN CONTROL TABLE (uncomment when test passes)
    html_rows = ""
    for idx, u in enumerate(pruned_utxos_global):
        checked = "checked"
        value = format_btc(u['value'])
        txid_short = u['txid'][:12] + "…" + u['txid'][-8:]
        confirmed = "Yes" if u.get('status', {}).get('confirmed', True) else "No"
        html_rows += f'''
        <tr>
            <td style="text-align:center;"><input type="checkbox" {checked} onchange="updateSelection()" data-idx="{idx}"></td>
            <td style="font-family:monospace;">{value}</td>
            <td style="font-family:monospace;">{txid_short}</td>
            <td style="text-align:center;">{u['vout']}</td>
            <td style="text-align:center;">{confirmed}</td>
        </tr>'''

    table_html = textwrap.dedent(f'''
        <div style="max-height:520px; overflow-y:auto; border:2px solid #f7931a; border-radius:12px;">
        <table style="width:100%; border-collapse:collapse; background:#111; color:white;">
            <thead style="position:sticky; top:0; background:#f7931a; color:black;">
                <tr>
                    <th style="padding:12px;">Include</th>
                    <th style="padding:12px;">Value (sats)</th>
                    <th style="padding:12px;">TXID</th>
                    <th style="padding:12px;">vout</th>
                    <th style="padding:12px;">Confirmed</th>
                </tr>
            </thead>
            <tbody>{html_rows}</tbody>
        </table>
        </div>
        <script>
        const utxos = {json.dumps(pruned_utxos_global)};
        function updateSelection() {{
            const checked = Array.from(document.querySelectorAll('input[type=checkbox]:checked'))
                            .map(cb => parseInt(cb.dataset.idx));
            const selected = checked.map(i => utxos[i]);
            const total = selected.reduce((a,b) => a + b.value, 0);
            document.getElementById('selected-summary').innerHTML = 
                `<b>Selected:</b> ${checked.length} UTXOs • <b>Total:</b> ${total.toLocaleString()} sats`;
            const s = document.querySelector('[data-testid="state"]');
            if (s?.__gradio_internal__) s.__gradio_internal__.setValue(selected);
        }}
        updateSelection();
        </script>
        <div id="selected-summary" style="margin-top:12px; font-size:18px; color:#f7931a;"></div>
    ''').strip()
    """

    # Final debug print
    print(f"RETURNING table_html length: {len(table_html)}")
    print(f"pruned_utxos_global length: {len(pruned_utxos_global)}")

    return (
        f"""
        <div style="text-align:center; padding:20px;">
            <b style="font-size:22px; color:#f7931a;">Analysis Complete</b><br><br>
            Found <b>{len(utxos):,}</b> UTXOs • Keeping <b>{keep}</b> largest<br>
            <b style="color:#f7931a;">Strategy:</b> {strategy_name} • Format: <b>{detected}</b><br>
            {nuclear_warning}
            <br><br>
            Click <b>Generate Transaction</b> to continue
        </div>
        """,
        gr.update(visible=True),     # generate_row
        gr.update(visible=True),     # coin_control_row
        table_html,                  # coin_table_html (gr.HTML)
        pruned_utxos_global          # selected_utxos_state (raw list)
    )
