import torch
import torch.multiprocessing as mp
import torch.distributed as dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def communication(rank, m_array):
    msg_tensor1 = torch.from_numpy(m_array).clone().to(device)
    if rank == 0:
        # Send the tensor to process 1
        dist.send(tensor=msg_tensor1, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=msg_tensor1, src=0)

    msg_tensor0 = torch.from_numpy(m_array).clone().to(device)
    if rank == 1:
        # Send the tensor to process 1
        dist.send(tensor=msg_tensor0, dst=0)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=msg_tensor0, src=1)
    return msg_tensor1.cpu().data.numpy().flatten() if rank == 1 else msg_tensor0.cpu().data.numpy().flatten()
