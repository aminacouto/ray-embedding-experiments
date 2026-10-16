## Configuração do Ambiente Virtual (.venv)

Siga os passos abaixo para configurar o ambiente de desenvolvimento local. Certifique-se de ter o **Python 3.x** instalado em sua máquina.

### 1. Clonar o repositório e acessar a pasta
Abra o terminal e execute os comandos:
```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd <NOME_DA_PASTA_DO_PROJETO>
```

### 2. Criar o Ambiente Virtual
Crie um ambiente isolado chamado `.venv-ray-test` na raiz do projeto:
```bash
python -m venv .venv-ray-test
```
*(Nota: Em algumas distribuições Linux, pode ser necessário instalar o pacote primeiro via `sudo apt install python3-venv`).*


### 3. Ativar o Ambiente Virtual
Ative o ambiente de acordo com o seu sistema operacional:

* **Windows (Prompt de Comando / CMD):**
  ```cmd
  .venv-ray-test\Scripts\activate.bat
  ```
* **Windows (PowerShell):**
  ```powershell
  .venv-ray-test\Scripts\Activate.ps1
  ```
  *(Se receber um erro de script bloqueado, execute antes: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)*
* **macOS / Linux (Bash ou Zsh):**
  ```bash
  source .venv-ray-test/bin/activate
  ```

> **Como saber se deu certo?** O nome `(.venv-ray-test)` aparecerá no início da linha do seu terminal.

### 4. Atualizar o Gerenciador de Pacotes (Opcional, mas recomendado)
Com o ambiente ativado, atualize o `pip`:
```bash
python -m pip install --upgrade pip
```

### 5. Instalar as Dependências do Projeto
Instale todas as bibliotecas necessárias listadas no arquivo `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Como Desativar o Ambiente
Quando terminar de trabalhar no projeto e quiser sair do ambiente virtual, basta digitar:
```bash
deactivate
```