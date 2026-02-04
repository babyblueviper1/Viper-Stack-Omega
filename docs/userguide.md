# Ωmega Pruner User Guide

## 1. Introduction

**Ωmega Pruner** is a fee-aware UTXO consolidation tool designed for non-custodial, single-sig Bitcoin users. It helps users gain full control over their coins by consolidating multiple UTXOs (Unspent Transaction Outputs) into a smaller set, minimizing transaction fees and improving wallet efficiency. With its live network context, deterministic selection model, and clear warnings around **CIOH** (Chain Identity Overlap & Linkage), Ωmega Pruner empowers users to make informed decisions and take ownership of their Bitcoin privacy and sovereignty.

### Key Features

- Fee-aware UTXO consolidation
- Deterministic transaction selection for predictable results
- Privacy-conscious design, with a focus on minimizing CIOH risk
- Non-custodial PSBT (Partially Signed Bitcoin Transaction) generation
- Live mempool insights and fee analysis for dynamic decision-making

## 2. Getting Started

### Accessing Ωmega Pruner

- Visit the Ωmega Pruner website.
- Make sure you’re using a supported browser (Google Chrome, Firefox, etc.).
- No registration is required; simply navigate to the tool’s interface.

### What You’ll Need

- A Bitcoin wallet with multiple UTXOs to consolidate (e.g., hardware wallet, software wallet).
- Basic familiarity with how Bitcoin works and the concept of UTXOs.

Ωmega Pruner is a single-sig demo tool, meaning it consolidates UTXOs from a single Bitcoin address and generates a PSBT that can be signed and broadcast using any compatible wallet.

## 3. Interface Walkthrough

The Ωmega Pruner interface is designed to be intuitive and minimalistic. Here’s a walkthrough of the key components:

### UTXO Display
Upon entering your address, the tool will enumerate all the UTXOs associated with it. For each UTXO, you’ll see:

- **Value**: The amount in satoshis.
- **Age**: How long it has been since the UTXO was created.
- **Script Type**: Identifying the script type (e.g., SegWit, Taproot).
- **Weight**: The UTXO’s size in weight units (important for fee optimization).

### Fee Context Layer
This section provides live, real-time fee data:

- **Current Fee** (economy, 1h, 30m): Provides a snapshot of current network conditions.
- **Mempool Status**: This shows the backlog of unconfirmed transactions and the current block height.
- **Fee Medians**: Compare the current fee against the 1-day, 1-week, and 1-month median values to assess whether it’s an optimal time for consolidation.

### Consolidation Actions

- **Selection & Consolidation Policy**: Ωmega Pruner will automatically select the best UTXOs to consolidate, based on a deterministic algorithm. You can also manually adjust these selections if needed.
- **CIOH Risk**: If you’re consolidating UTXOs that could reveal patterns or links to other addresses, this risk is highlighted.
- **PSBT Generation**: When you’re ready, generate a Partially Signed Bitcoin Transaction (PSBT) that is ready for signing in your hardware or software wallet.

## 4. Step-by-Step Tutorial

Here’s how to consolidate your UTXOs using Ωmega Pruner:

1. **Enter Your Bitcoin Address**  
   Simply input a single Bitcoin address into the tool. Ωmega Pruner will load all available UTXOs associated with that address.  
   **Note**: Only one address per run is allowed — this helps prevent cross-wallet mixing and maintains security.

2. **Review the UTXOs**  
   Assess the UTXOs displayed: Review the value, age, script type, and weight of each UTXO.  
   The interface will also show a fee context comparison to help you decide if consolidation is optimal based on current network conditions.

3. **Select UTXOs for Consolidation**  
   The tool will automatically suggest UTXOs for consolidation. You can also manually select which UTXOs to consolidate based on your preferences.  
   The selections are deterministic, so the same inputs will generate the same PSBT every time.

4. **Generate the PSBT**  
   Once you’ve selected your UTXOs, click “Generate PSBT”. This will create a PSBT that you can download and sign using your preferred Bitcoin wallet.  
   **Note**: The PSBT is unsigned — you’ll need a compatible wallet to sign and broadcast it.

## 5. Common Questions & Troubleshooting

- **What should I do if I see legacy addresses (1… or 3…)?**  
  Legacy and nested SegWit addresses are displayed for transparency, but they cannot be consolidated through Ωmega Pruner. You should spend or migrate these UTXOs separately before attempting consolidation.

- **I see Taproot inputs, but the tool won’t generate a PSBT. Why?**  
  Taproot inputs require specific derivation paths for signing in certain hardware wallets. If no path is detected, you’ll receive a non-blocking warning. To proceed, import the PSBT into a wallet that supports Taproot or recreate the transaction there.

- **What’s CIOH and why does it matter?**  
  CIOH (Chain Identity Overlap & Linkage) refers to the risk of linking multiple UTXOs or transactions to the same identity. Ωmega Pruner displays warnings if consolidating UTXOs could expose your transaction history or link different addresses.

## 6. Best Practices & Recommendations

### When to Consolidate

- Consolidate when network fees are low or when you have many small UTXOs, which are considered dust.
- Avoid consolidating during periods of high fee volatility unless necessary.

### Security Tips

- Always use Ωmega Pruner on a secure device.
- Remember, Ωmega Pruner does not sign or broadcast transactions — it only generates PSBTs. Always double-check transactions before signing.

## 7. Conclusion

Ωmega Pruner is a powerful tool for Bitcoin users who want to consolidate their UTXOs and take full control over their coin management. By using this guide, you’ve learned how to consolidate your UTXOs, understand network conditions, and optimize your wallet’s fee efficiency.

### Next Steps

- Join the Ωmega Pruner community to stay updated on new features and updates.
- Share your feedback to help improve the tool.

## Support

**babyblueviperbusiness@gmail.com**
