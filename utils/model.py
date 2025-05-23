import torch
import torch.nn as nn
from transformers import (AutoModel,
                          AutoModelForCausalLM,
                          AutoTokenizer
                          )

class GraphQATransformer(nn.Module):
    """
    Querying transformer for graph q/a, like blip 2 but encodes graphs instead of images.
    Freezes the graph encoder and the causal LM; only trains the cross-modal Q-former.
    """

    def __init__(
            self,
            graph_encoder_name: str = "microsoft/graphormer-large",
            llm_name: str: "Llama-4-Scout-17B-16E-Instruct",
            qformer_hidden_size: int = 768,
            qformer_num_queries: int = 32,
            qformer_n_heads: int = 8,
            qformer_n_layers: int = 6,
            ):
        super().__init__()
            
        # load and freeze a pretrained graph encoder
        self.graph_encoder = AutoModel.from_pretrained(graph_encoder_name)
        for p in self.graph_encoder.parameters():
            p.requires_grad = False

        # load and freeze a pretrained causal lm (llama 4)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        for p in self.llm.parameters():
            p.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # q-former: learnable queries + transformer decoder for cross attention
        self.num_queries = qformer_num_queries
        self.query_embed = nn.Parameter(torch.randn(qformer_num_queries, qformer_hidden_size))
        decoder_layer = nn.TransformerDecoderLayer(
                d_model=q_former_hidden_size,
                nhead=qformer_n_heads,
                dim_feedforward=qformer_hidden_size * 4,
                dropout=0.1,
                activation="gelu"
                )
        # TransformerDecoder is a stack of decoder layers
        self.qformer = nn.TransformerDecoder(decoder_layer, num_layers=qformer_n_layers)

        # project to llm embedding dimension (if different)
        graph_hidden = self.graph_encoder.config.hidden_size
        llm_hidden = self.llm.config.n_embd
        self.graph_proj = nn.Linear(graph_hidden, qformer_hidden_size) if graph_hidden != qformer_hidden_size else nn.Identity()
        self.qformer_proj = nn.Liner(qformer_hidden_size, llm_hidden) if qformer_hidden_size != llm_hidden else nn.Identity()

    def forward(
            self,
            node_feat: torch.Tensor,
            attn_mask: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels=None
            ):

        # encode the graph
        # node_feat: [B, N, F]; attn_mask: [B, N]
        g_out = self.graph_encoder(node_feat=node_feat, attn_mask=attn_mask) 
        graph_feats = g_out.last_hidden_state # [B, N, graph_hidden]
        graph_feats = self.graph_proj(graph_feats) # [B, N, q_hidden]
        memory = graph_feats.permute(1, 0, 2) # [N, B, H]

        # q-former queries
        batch_size = input_ids.size(0)
        queries = self.query_embed.unsqueeze(1).expand(-1, batch_size, -1)
        q_feat = self.qformer(tgt=queries, memory=memory)
        q_feat = q_feat.permute(1, 0, 2)
        prefix_embeds = self.qformer_proj(q_feat)

        # text embeddings
        text_embeds = self.llm.transformer.wte(input_ids)

        # concat and build full attn mask
        inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)
        prefix_mask = torch.ones(batch_size, self.num_queries, device=attn_mask.device, dtype=attention_mask.dtype)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # forward through llm
        outputs = self.llm(
                inputs_embeds,
                attention_mask=full_mask,
                labels=labels,
                return_dict=True,
                )
        return outputs
