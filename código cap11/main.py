from redes_neuronales import red_neuronal

def mostrar_menu(opciones):
    print('Seleccione una opción:')
    for clave in sorted(opciones):
        print(f' {clave}) {opciones[clave][0]}')


def leer_opcion(opciones):
    while (a := input('Opción: ')) not in opciones:
        print('Opción incorrecta, vuelva a intentarlo.')
    return a


def ejecutar_opcion(opcion, opciones):
    opciones[opcion][1]()


def generar_menu(opciones, opcion_salida):
    opcion = None
    while opcion != opcion_salida:
        mostrar_menu(opciones)
        opcion = leer_opcion(opciones)
        ejecutar_opcion(opcion, opciones)
        print()


def menu_principal():
    print('¿Cuántas capas quieres que tenga la red neuronal?')
    opciones = {
        '1': ('12 capas', accion1),
        '2': ('24 capas', accion2),
        '3': ('26 capas', accion3),
        '4': ('31 capas', accion4),
        '5': ('Salir', salir)
    }
    generar_menu(opciones, '5')


def accion1():
    print('Has elegido la opción con 12 capas')
    red_neuronal.main(12)


def accion2():
    print('Has elegido la opción con 24 capas')
    red_neuronal.main(24)


def accion3():
    print('Has elegido la opción con 26 capas')
    red_neuronal.main(26)

def accion4():
    print('Has elegido la opción con 31 capas')
    red_neuronal.main(311)


def salir():
    print('Saliendo')


if __name__ == '__main__':
    menu_principal()