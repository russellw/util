package aklo;

import static org.objectweb.asm.Opcodes.*;

import java.util.Arrays;
import java.util.Map;
import org.objectweb.asm.MethodVisitor;

final class Call extends Nary {
  Call(Object... args) {
    super(args);
  }

  @Override
  Type type() {
    var f = get(0);
    if (f instanceof Fn f1) return f1.rtype;
    return Type.OBJECT;
  }

  @Override
  void emit(Map<Object, Integer> refs, MethodVisitor mv) {
    var f = get(0);
    if (f instanceof Fn f1) {
      for (var i = 1; i < size(); i++) load(refs, mv, get(i));
      mv.visitMethodInsn(INVOKESTATIC, "a", f1.name, f1.descriptor(), false);
      return;
    }
    for (var i = 0; i < size(); i++) load(refs, mv, get(i));
    switch (size() - 1) {
      case 1 -> mv.visitMethodInsn(
          INVOKEINTERFACE,
          "java/util/function/Function",
          "apply",
          "(Ljava/lang/Object;)Ljava/lang/Object;",
          true);
      case 2 -> mv.visitMethodInsn(
          INVOKEINTERFACE,
          "java/util/function/BiFunction",
          "apply",
          "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
          true);
      case 3 -> mv.visitMethodInsn(
          INVOKEINTERFACE,
          "aklo/TriFunction",
          "apply",
          "(Ljava/lang/Object;Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
          true);
      default -> throw new IllegalArgumentException(this + Arrays.toString(args));
    }
  }
}
