package aklo;

import static org.objectweb.asm.Opcodes.RETURN;

import java.util.Map;
import org.objectweb.asm.MethodVisitor;

final class ReturnVoid extends Instruction {
  @Override
  boolean isTerminator() {
    return true;
  }

  @Override
  void emit(Map<Object, Integer> refs, MethodVisitor mv) {
    mv.visitInsn(RETURN);
  }
}
