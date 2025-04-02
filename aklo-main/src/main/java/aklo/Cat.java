package aklo;

import static org.objectweb.asm.Opcodes.*;

import java.util.Map;
import org.objectweb.asm.MethodVisitor;

final class Cat extends Binary {
  Cat(Object arg0, Object arg1) {
    super(arg0, arg1);
  }

  @Override
  void emit(Map<Object, Integer> refs, MethodVisitor mv) {
    load(refs, mv, arg0);
    load(refs, mv, arg1);
    mv.visitMethodInsn(
        INVOKESTATIC,
        "aklo/Etc",
        "cat",
        "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/util/List;",
        false);
  }
}
