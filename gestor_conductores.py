import os
import sys
import json
import subprocess
import urllib.request
import tkinter as tk
from tkinter import ttk, messagebox
import db
import config
import sv_ttk

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable 


class Gestor(tk.Tk):
    def __init__(self):
        super().__init__()
        db.init()
        self.title("Gestor de Conductores")
        self.geometry("640x600")
        self.minsize(580, 540)
        sv_ttk.set_theme("light")                 # aspecto Windows 11 ("dark" = modo oscuro)
        self.configure(padx=20, pady=16)

        # --- Encabezado: título a la izquierda, botón principal a la derecha ---
        cab = ttk.Frame(self)
        cab.pack(fill="x")
        textos = ttk.Frame(cab)
        textos.pack(side="left")
        ttk.Label(textos, text="Detección de Somnolencia", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(textos, text="Registrá a los conductores y arrancá el monitoreo.",
                  foreground="#666").pack(anchor="w")
        ttk.Button(cab, text="▶  Iniciar sistema", style="Accent.TButton",
                   command=self.iniciar).pack(side="right", ipadx=10, ipady=6)

        # --- Tarjeta: nuevo conductor ---
        nuevo = ttk.LabelFrame(self, text=" Nuevo conductor ", padding=14)
        nuevo.pack(fill="x", pady=(18, 10))
        ttk.Label(nuevo, text="Nombre").grid(row=0, column=0, sticky="w", pady=4)
        self.ent_nombre = ttk.Entry(nuevo, width=32)
        self.ent_nombre.grid(row=0, column=1, sticky="w", padx=10)
        ttk.Label(nuevo, text="Chat de Telegram").grid(row=1, column=0, sticky="w", pady=4)
        self.ent_contacto = ttk.Entry(nuevo, width=32)
        self.ent_contacto.grid(row=1, column=1, sticky="w", padx=10)
        ttk.Button(nuevo, text="🔎  Buscar", command=self.obtener_chatid).grid(row=1, column=2, sticky="w")
        
        ttk.Label(nuevo, text="Se abre la cámara: mové un poco la cabeza hasta que junte las 40 fotos.",
                  foreground="#666").grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.btn_agregar = ttk.Button(nuevo, text="📷  Agregar y capturar rostro", style="Accent.TButton",
                                      command=self.agregar)
        self.btn_agregar.grid(row=2, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self.ent_nombre.bind("<Return>", lambda e: self.agregar())
        self.ent_contacto.bind("<Return>", lambda e: self.agregar())

        # --- Tarjeta: lista de conductores ---
        lista = ttk.LabelFrame(self, text=" Conductores registrados ", padding=14)
        lista.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(lista, columns=("nombre", "telegram", "fotos"),
                                 show="headings", height=7)
        for col, txt, w, ali in [("nombre", "Nombre", 240, "w"),
                                 ("telegram", "Telegram", 160, "center"),
                                 ("fotos", "Fotos", 80, "center")]:
            self.tree.heading(col, text=txt, anchor=ali)
            self.tree.column(col, width=w, anchor=ali)
        self.tree.pack(fill="both", expand=True)

        botones = ttk.Frame(lista)
        botones.pack(fill="x", pady=(10, 0))
        self.btn_recargar = ttk.Button(botones, text="📷  Recargar rostro", command=self.recargar)
        self.btn_recargar.pack(side="left")
        ttk.Button(botones, text="🗑  Eliminar", command=self.eliminar).pack(side="left", padx=8)
        self.estado = ttk.Label(self, text="Listo.", foreground="#666")
        self.estado.pack(anchor="w", pady=(10, 0))  

        self.refrescar()


    def refrescar(self):
        self.tree.delete(*self.tree.get_children())
        for nombre, chat, n in db.listar():
            # iid = nombre real de la base (minúsculas); lo que se ve, con mayúscula inicial
            self.tree.insert("", "end", iid=nombre,
                             values=(nombre.title(), "✔ Configurado" if chat else "—", n))
    def avisar(self, texto):
        self.estado.config(text=texto)

    def fotos_de(self, nombre):
        for n, _, fotos in db.listar():
            if n == nombre:
                return fotos
        return 0

    def agregar(self):
        nombre = self.ent_nombre.get().strip().lower()
        contacto = self.ent_contacto.get().strip()
        if not nombre:
            self.avisar("⚠  Escribí un nombre para el conductor.")
            self.ent_nombre.focus()
            return
        if self.tree.exists(nombre):               # ya está en la lista
            if not messagebox.askyesno("Ya existe",
                                       f"{nombre.title()} ya está registrado con {self.fotos_de(nombre)} fotos.\n"
                                       "¿Querés agregarle fotos nuevas?"):
                return
        self.capturar(nombre, contacto)
        self.ent_nombre.delete(0, "end")
        self.ent_contacto.delete(0, "end")

    def recargar(self):
        sel = self.tree.selection()
        if not sel:
            self.avisar("⚠  Elegí un conductor de la lista para recargar su rostro.")
            return
        self.capturar(sel[0], "")                  # sin contacto: se mantiene el de la base

    def capturar(self, nombre, contacto):
        self.antes = self.fotos_de(nombre)         # para saber cuántas fotos se agregaron
        self.capturando = nombre
        self.proc = subprocess.Popen([PY, "cargar_conductor.py", "--nombre", nombre, "--contacto", contacto],
                                     cwd=BASE, creationflags=subprocess.CREATE_NEW_CONSOLE)
        self.btn_agregar.state(["disabled"])       # no abrir la cámara dos veces
        self.btn_recargar.state(["disabled"])
        self.avisar(f"📷  Capturando el rostro de {nombre.title()}... mirá a la cámara.")
        self.after(500, self.esperar_captura)

    def esperar_captura(self):
        if self.proc.poll() is None:               # sigue abierta la cámara: volver a mirar en 0,5 s
            self.after(500, self.esperar_captura)
            return
        self.btn_agregar.state(["!disabled"])
        self.btn_recargar.state(["!disabled"])
        self.refrescar()
        nuevas = self.fotos_de(self.capturando) - self.antes
        if nuevas > 0:
            self.avisar(f"✔  {self.capturando.title()}: se guardaron {nuevas} fotos.")
        else:
            self.avisar(f"⚠  No se guardó ninguna foto de {self.capturando.title()}. ¿Está conectada la cámara?")       

 

    def eliminar(self):
        sel = self.tree.selection()
        if not sel:
            return
        nombre = sel[0]                          # el iid es el nombre real de la base
        if messagebox.askyesno("Eliminar", f"¿Eliminar a '{nombre}'?"):
            db.eliminar(str(nombre))
            self.refrescar()
            self.avisar(f"🗑  Se eliminó a {nombre.title()}.")

    def obtener_chatid(self):
        if not config.TG_TOKEN:
            messagebox.showerror("Falta el token", "No está configurada la variable de entorno TG_TOKEN.\n"
                                                   "Ver el README para configurarla.")
            return
        self.avisar("🔎  Buscando contactos en Telegram...")
        self.update_idletasks()                    # que se vea el aviso antes de esperar a Telegram
        try:
            url = f"https://api.telegram.org/bot{config.TG_TOKEN}/getUpdates"
            data = json.loads(urllib.request.urlopen(url, timeout=10).read().decode())
        except Exception:
            self.avisar("⚠  No se pudo conectar con Telegram. ¿Hay internet?")
            return
        ids = {}
        for u in data.get("result", []):
            m = u.get("message") or u.get("edited_message") or {}
            ch = m.get("chat", {})
            if ch:
                ids[ch.get("id")] = f"{ch.get('first_name', '')} {ch.get('last_name', '') or ''}".strip()
        if not ids:
            self.avisar("⚠  Nadie le escribió al bot todavía: pedile al contacto que le envíe /start.")
            return
        self.avisar("Elegí el contacto de la lista.")
        self.elegir_contacto(ids)

    def elegir_contacto(self, ids):
        v = tk.Toplevel(self)                      # ventanita aparte, encima del gestor
        v.title("Elegir contacto")
        v.configure(padx=14, pady=14)
        v.transient(self)
        v.grab_set()                               # bloquea el gestor hasta cerrarla
        ttk.Label(v, text="¿Quién va a recibir las alertas?", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        lista = ttk.Treeview(v, columns=("nombre",), show="", height=min(len(ids), 6))
        for chat, nombre in ids.items():
            lista.insert("", "end", iid=str(chat), values=(nombre or "(sin nombre)",))
        lista.pack(fill="x", pady=10)
        lista.selection_set(lista.get_children()[0])

        def usar(_=None):
            self.ent_contacto.delete(0, "end")
            self.ent_contacto.insert(0, lista.selection()[0])
            nombre = lista.item(lista.selection()[0])["values"][0]
            self.avisar(f"✔  Se completó el chat de {nombre}.")
            v.destroy()

        lista.bind("<Double-1>", usar)             # doble clic también elige
        ttk.Button(v, text="Usar este contacto", style="Accent.TButton", command=usar).pack(anchor="e")        


    def iniciar(self):
        subprocess.Popen([PY, "sistema.py"], cwd=BASE, creationflags=subprocess.CREATE_NEW_CONSOLE)

        self.avisar("▶  Iniciando el sistema en otra ventana...")


if __name__ == "__main__":

    Gestor().mainloop()
