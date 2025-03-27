# DeepRiverDivers
Exploitative Poker Agent maximizing river decisions

# Model Architecture
Action Sequence Encoder
Each action is represented using the following components:
	•	Actor: binary flag (hero or villain)
	•	Action Type: 5-way embedding mapped to 4 dimensions (fold, check, call, raise, bet)
	•	Bet Size Bucket: 6-dimensional one-hot vector (0-20%, 20-50%, 50-100%, 100-200%, >200%)
	•	Street: 4-way embedding mapped to 4 dimensions (preflop, flop, turn river)
	•	Position: 6-way embedding mapped to 4 dimensions (LJ, HJ, CO, BU, SB, BB)
These are concatenated into a 19-dimensional vector per action, resulting in shape (B, T, 19). The sequence is passed through a multilayer perceptron (MLP): 19 → 32 → ReLU → 32, giving output of shape (B, T, 32). Positional embeddings are added to preserve sequence order. The output is then passed through a TransformerEncoder (self-attention) to model dependencies across the action sequence.
Hand Evaluator Encoder
Each of the seven known cards (2 hole cards + 5 board cards) is represented using:
	•	Rank: 13-way embedding to 8 dimensions (2, 3, 4, … Q, K, A)
	•	Suit: 4-way embedding to 4 dimensions (Spade, Heart, Diamond, Club)
	•	Street: 4-way embedding to 4 dimensions (hole cards, flop cards, turn card, river card)
These features are concatenated into a 16-dimensional vector per card, resulting in shape (B, 7, 16). This is passed through an MLP: 16 → 24 → ReLU → 32, producing (B, 7, 32). Positional embeddings are added to provide ordering information across the cards. The sequence is then passed through a TransformerEncoder (self-attention) to model interactions between the cards (e.g., pairs, flushes, blockers).
Cross-Attention and Fusion
Cross-attention is applied in both directions:
	•	Action tokens attend to card tokens
	•	Card tokens attend to action tokens
This allows each stream to incorporate context from the other. After cross-attention, mean pooling is applied across the time dimension:
	•	Action sequence encoder output (B, T, 32) is pooled to (B, 32)
	•	Hand evaluator encoder output (B, 7, 32) is pooled to (B, 32)
The two context vectors are concatenated to produce a fused vector of shape (B, 64).
Output Head
The fused context vector is passed through a final MLP to produce a scalar EV prediction:
	•	64 → 64 → ReLU → Dropout
	•	64 → 32 → ReLU → Dropout
	•	32 → 1 (output EV)

Training Procedure
Phase 1: GTO Pretraining
The model is first trained on solver-generated data using the ground-truth EV from the GTO engine. Mean Squared Error (MSE) is used as the loss function:
loss = MSE(predicted_ev, gto_ev)
This phase provides a strong baseline by learning from theoretically optimal decisions.
Phase 2: Human Data Fine-Tuning
The model is fine-tuned using real human hand histories. Target EV is calculated as follows:
	•	If the player called and won: EV = pot_size - call_amount
	•	If the player called and lost: EV = -call_amount
	•	If the player folded: EV = 0 (or left out if modeling only calls)
To stabilize training and reduce variance, EVs are scaled by a factor of 1/10:
scaled_ev = original_ev / 10
MSE loss is used during training:
loss = MSE(predicted_ev, scaled_target_ev)

This two-stage approach allows the model to first learn ideal play, then adapt to human patterns and variance. 
