from Bio import Entrez
import os

# 设置你的邮箱（NCBI要求）
Entrez.email = ""

# 缓存文件路径
CACHE_FILE = "./gene_sequence_cache.txt"


def load_gene_cache():
    """加载缓存文件，返回一个字典 {基因名称: 序列}"""
    if not os.path.exists(CACHE_FILE):
        return {}
    gene_cache = {}
    with open(CACHE_FILE, "r") as f:
        for line in f:
            gene_name, sequence = line.strip().split("\t")
            gene_cache[gene_name] = sequence
    return gene_cache


def save_gene_cache(gene_cache):
    """将缓存字典保存到文件"""
    with open(CACHE_FILE, "w") as f:
        for gene_name, sequence in gene_cache.items():
            f.write(f"{gene_name}\t{sequence}\n")


def fetch_gene_sequence(gene_name, cache):
    """获取基因的序列（cDNA），优先从缓存中读取"""
    # 如果缓存中已存在，直接返回
    if gene_name in cache:
        # print(f"从缓存中获取 {gene_name} 的序列")
        return cache[gene_name]
    else:
        # 所有能找到的序列都已经在缓存里了
        return None

    # 搜索Gene数据库获取Gene ID
    search_handle = Entrez.esearch(db="gene", term=f'"{gene_name}"[Gene]')
    search_results = Entrez.read(search_handle)
    search_handle.close()

    if not search_results["IdList"]:
        # print(f"未找到基因 {gene_name}")
        return None
    gene_id = search_results["IdList"][0]

    link_handle = Entrez.elink(dbfrom="gene", db="nuccore", id=gene_id, linkname="gene_nuccore_refseqrna")
    link_results = Entrez.read(link_handle)
    link_handle.close()

    if not link_results[0].get("LinkSetDb"):
        # print(f"未找到 {gene_name} 的序列")
        return None
    nuccore_ids = [link["Id"] for link in link_results[0]["LinkSetDb"][0]["Link"]]
    if not nuccore_ids:
        # print(f"未找到 {gene_name} 的序列")
        return None

    # 获取第一个序列
    mrna_id = nuccore_ids[0]
    fetch_handle = Entrez.efetch(db="nuccore", id=mrna_id, rettype="fasta", retmode="text")
    fasta_data = fetch_handle.read()
    fetch_handle.close()

    # 提取纯碱基序列（去掉注释和换行）
    sequence = "".join(fasta_data.split("\n")[1:])

    # 将结果存入缓存
    cache[gene_name] = sequence
    save_gene_cache(cache)

    # print(f"成功获取 {gene_name} 的序列")
    return sequence


# gene_cache = load_gene_cache()  # 加载缓存
# gene_name = "PDCD2"
# sequence = fetch_gene_sequence(gene_name, gene_cache)
#
# if sequence:
#     print(f"{gene_name} 的序列为: {sequence}")
# else:
#     print(f"未找到 {gene_name} 的序列")



